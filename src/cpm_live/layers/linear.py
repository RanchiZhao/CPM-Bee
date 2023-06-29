# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import bmtrain as bmt
import math
import torch.nn.functional as F
import bitsandbytes as bnb
import bmtrain as bmt
from typing import TypeVar,overload,Optional,Union
from torch import Tensor, device, dtype
T = TypeVar("T", bound="torch.nn.Module")

class Linear(bmt.DistributedModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dtype: torch.dtype = torch.half,
        init_mean: float = 0.0,
        init_std: float = 1,
        scale_before: bool = False,
    ):
        super().__init__()
        self.dim_in = self.in_features = dim_in
        self.dim_out = self.out_features = dim_out
        self.scale_before = scale_before

        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(
                torch.nn.init.normal_, mean=init_mean, std=init_std
            ),
        )
        print(isinstance(self.weight, bmt.DistributedModule))
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): The input of linear layer
        Returns:
            :obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``: The output of the linear transform y.
        """  # noqa: E501
        if self.scale_before:
            x = x / math.sqrt(self.dim_in)
            x = F.linear(x, self.weight)
        else:
            x = F.linear(x, self.weight)
            x = x / math.sqrt(self.dim_in)
        return x
    
class Linear4bit(Linear):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        compute_dtype: torch.dtype = None,
        compress_statistics: bool = True,
        quant_type: str = 'fp4',
        init_mean: float = 0.0,
        init_std: float = 1,
    ):
        super().__init__(dim_in, dim_out)
        self.weight = Params4bit(
            self.weight.data, 
            requires_grad=False, 
            compress_statistics=compress_statistics, 
            quant_type=quant_type, 
            init_method=bmt.ParameterInitializer(
                torch.nn.init.normal_, 
                mean=init_mean, 
                std=init_std
            ),
        )
        print(type(self.weight))
        print(self.weight.dtype)
        print(self.quant_state)
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor):
        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if getattr(self.weight, 'quant_state', None) is None:
            print('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')
        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)
        out = bnb.matmul_4bit(x, self.weight.t(), bias=None, quant_state=self.weight.quant_state)
        out = out.to(inp_dtype)
        return out

from typing import Callable, Optional
from bmtrain.utils import round_up
from bmtrain.global_var import config
from bmtrain.parameter import OpAllGather

class Params4bit(bmt.DistributedParameter):
    _original_shape : torch.Size
    _start_partition : int
    _end_partition : int
    _init_method : Optional[Callable[['bmt.DistributedParameter'], None]]
    _in_checkpoint_block : bool
    _group : Optional[str]
    def __new__(cls, 
                data=None, 
                requires_grad=True, 
                quant_state=None, 
                blocksize=64, 
                compress_statistics=True, 
                quant_type='fp4',
                init_method : Optional[Callable[['bmt.DistributedParameter'], None]] = None,
                group : Optional[str] = None):

        if not config["initialized"]:
            raise RuntimeError("BMTrain is not initialized")

        num_of_elements = data.numel()

        cuda_tensor = torch.tensor([], dtype=data.dtype, device="cuda") 
        cuda_storage_size = round_up(num_of_elements, config["world_size"]) // config["world_size"]

        original_shape = data.size()

        cuda_storage = cuda_tensor.storage_type()(cuda_storage_size)

        start_of_partition = cuda_storage_size * config["rank"]
        end_of_partition = min(num_of_elements, cuda_storage_size * (config["rank"] + 1))

        # FX: cuda_tensor_size < 0 if num_of_elements is too small
        cuda_tensor_size = max(end_of_partition - start_of_partition, 0)

        cuda_tensor.set_(cuda_storage, 0, (cuda_tensor_size,))
        cuda_tensor.copy_(data.view(-1)[start_of_partition: end_of_partition])
        # ret = torch.Tensor._make_subclass(cls, cuda_tensor, requires_grad)
        
        ret = torch.Tensor._make_subclass(cls, cuda_tensor, requires_grad)
        print(isinstance(ret,Params4bit))
        ret.blocksize = blocksize
        ret.compress_statistics = compress_statistics
        ret.quant_type = quant_type
        ret.quant_state = quant_state
        w = data.contiguous().half()
        w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=ret.blocksize, compress_statistics=ret.compress_statistics, quant_type=ret.quant_type)
        ret.data = w_4bit
        print(ret.data.shape)
        ret.quant_state = quant_state

        setattr(ret, "_original_shape", original_shape)
        setattr(ret, "_start_partition", start_of_partition)
        setattr(ret, "_end_partition", end_of_partition)
        setattr(ret, "_init_method", init_method)
        setattr(ret, "_in_checkpoint_block", False)
        setattr(ret, "_group", group)
        print(isinstance(ret,Params4bit))
        return ret
        
    @property
    def group(self):
        """The group name of the distributed parameter."""

        return self._group

    def gather(self) -> torch.Tensor:
        """Gather the data from all the distributed nodes.

        Return:
            torch.Tensor: The gathered data.
        
        """
        with torch.cuda.stream(config['load_stream']):
            output_tensor = OpAllGather.apply(self)
        current_stream = torch.cuda.current_stream()
        output_tensor.record_stream( current_stream )
        current_stream.wait_stream(config['load_stream'])
        return output_tensor

    def _copy_data(self, data : torch.Tensor):
        self.data.copy_(data.view(-1)[self._start_partition : self._end_partition])
    
        # if data is None:
        #     data = torch.empty(0)

        # obj = super().__new__(cls, data, requires_grad)
        # obj.blocksize = blocksize
        # obj.compress_statistics = compress_statistics
        # obj.quant_type = quant_type
        # obj.quant_state = quant_state
        # w = data.contiguous().half()
        # w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=obj.blocksize, compress_statistics=obj.compress_statistics, quant_type=obj.quant_type)
        # obj.data = w_4bit
        # obj.quant_state = quant_state
        # return obj

    # def __init__(self, data=None, requires_grad=True, quant_state=None, blocksize=64, compress_statistics=True, quant_type='fp4'):
    #     super().__init__()
    #     self._in_checkpoint_block = False
    #     self._original_shape = data.shape
    #     self._start_partition = 0  
    #     self._end_partition = 0  
    #     self._init_method = None  
    #     self._in_checkpoint_block = False
    #     self._group = None 


class Params4bitDistributed(bmt.DistributedParameter):
    def __new__(cls, data=None, requires_grad=True, quant_state=None, blocksize=64, compress_statistics=True, quant_type='fp4', init_method=None, group=None):
        # 创建 Params4bit 实例
        params4bit = Params4bit(data, requires_grad, quant_state, blocksize, compress_statistics, quant_type)
        # 创建 DistributedParameter 实例，并将 Params4bit 的数据作为输入
        self = super(Params4bitDistributed, cls).__new__(cls, params4bit.data, requires_grad, init_method, group)
        # 将 Params4bit 的属性复制到新的实例中
        self.blocksize = params4bit.blocksize
        self.compress_statistics = params4bit.compress_statistics
        self.quant_type = params4bit.quant_type
        self.quant_state = params4bit.quant_state
        # 返回新的实例
        return self
