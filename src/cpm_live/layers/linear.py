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
    ):
        super().__init__(dim_in, dim_out, dtype=torch.float32)
        self.weight = Params4bit(self.weight.data, requires_grad=False, compress_statistics=compress_statistics, quant_type=quant_type)
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

class Params4bit(bmt.DistributedParameter):
    def __new__(cls, data=None, requires_grad=True, quant_state=None, blocksize=64, compress_statistics=True, quant_type='fp4'):
        if data is None:
            data = torch.empty(0)

        self = super().__new__(cls, data=data, requires_grad=requires_grad)
        # self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.data = data
        return self

    def cuda(self, device):
        w = self.data.contiguous().half().cuda(device)
        # print(w.shape) #torch.Size([1280, 4096])
        print("---")
        w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=self.blocksize, compress_statistics=self.compress_statistics, quant_type=self.quant_type)
        # print(w_4bit.shape) #torch.Size([2621440, 1])
        self.data = w_4bit
        self.quant_state = quant_state
        return self

    @overload
    def to(self: T, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ..., non_blocking: bool = ...,) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if (device is not None and device.type == "cuda" and self.data.device.type == "cpu"):
            return self.cuda(device)
        else:
            s = self.quant_state
            if s is not None:
                # make sure the quantization state is on the right device
                s[0] = s[0].to(device)
                if self.compress_statistics:
                    # TODO: refactor this. This is a nightmare
                    # for 4-bit: 
                    # state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
                    # state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
                    #s[-2][0] = s[-2][0].to(device) # offset
                    #s[-2][1][0] = s[-2][1][0].to(device) # nested absmax

                    # for 8-bit
                    s[-2][0] = s[-2][0].to(device) # offset
                    s[-2][1][0] = s[-2][1][0].to(device) # nested quantiation state statitics
                    s[-2][1][1] = s[-2][1][1].to(device) # nested quantiation codebook
            new_param = Params4bit(super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                                  requires_grad=self.requires_grad, quant_state=self.quant_state,
                                   blocksize=self.blocksize, compress_statistics=self.compress_statistics,
                                   quant_type=self.quant_type)

            return new_param
