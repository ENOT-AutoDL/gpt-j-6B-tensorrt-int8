from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


class TrtModel:
    """
    Wrapper class that initializes TensorRT engine and takes care of
    dynamic shapes processing.
    """

    def __init__(self, path_to_engine: str):
        """
        Parameters
        ----------
        path_to_engine : str
            Path to TensorRT engine file.

        """
        runtime = trt.Runtime(TRT_LOGGER)
        with open(path_to_engine, 'rb') as engine_file:
            self.engine = runtime.deserialize_cuda_engine(engine_file.read())

        self.context = self.engine.create_execution_context()

    @property
    def inputs(self) -> List[str]:
        """List of inputs names."""
        return [self.engine[i] for i in range(len(self.engine)) if self.engine.binding_is_input(i)]

    @property
    def outputs(self) -> List[str]:
        """List of outputs names."""
        return [self.engine[i] for i in range(len(self.engine)) if not self.engine.binding_is_input(i)]

    def binding_shape(self, name: str) -> Tuple[int, ...]:
        """
        Returns binding shape by binding name.

        Parameters
        ----------
        name : str
            Binding name.

        Returns
        -------
        Tuple[int, ...]
            Shape of the binding.

        """
        index = self.engine.get_binding_index(name)
        shape = self.context.get_binding_shape(index)
        return tuple(shape)

    def set_binding_shape(self, name: str, shape: Tuple[int, ...]) -> None:
        """
        Updates binding shape.

        Parameters
        ----------
        name : str
            Binding name.
        shape : Tuple[int, ...]
            Desired shape.

        """
        index = self.engine.get_binding_index(name)
        if not self.context.set_binding_shape(index, shape):
            raise RuntimeError(f'Cannot update binding shape (name = "{name}")')

    def binding_dtype(self, name: str) -> torch.dtype:
        """
        Returns binding data type by binding name.

        Parameters
        ----------
        name : str
            Binding name.

        Returns
        -------
        torch.dtype
            Binding data type.

        """
        trt_dtype = self.engine.get_binding_dtype(name)
        try:
            return {
                trt.int8: torch.int8,
                trt.int32: torch.int32,
                trt.float16: torch.float16,
                trt.float32: torch.float32,
                trt.bool: torch.bool,
            }[trt_dtype]
        except KeyError as exc:
            raise ValueError(f'Got unknown TensorRT dtype ({trt_dtype})') from exc

    def _prepare_output_tensor(self, name: str, output_tensor: Optional[torch.Tensor]) -> torch.Tensor:
        if output_tensor is None:
            output_tensor = torch.empty(
                size=self.binding_shape(name),
                dtype=self.binding_dtype(name),
                device='cuda',
            )
        else:
            output_tensor.resize_(*self.binding_shape(name))

        return output_tensor

    def _create_bindings(
        self,
        input_tensors: Dict[str, torch.Tensor],
        output_tensors_cache: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[List[int], Dict[str, torch.Tensor]]:
        if output_tensors_cache is None:
            output_tensors = {name: None for name in self.outputs}
        else:
            output_tensors = output_tensors_cache

        for name, data in input_tensors.items():
            self.set_binding_shape(name, data.shape)

        for name, data in output_tensors.items():
            output_tensors[name] = self._prepare_output_tensor(name, data)

        bindings: List = [None] * self.engine.num_bindings
        for name, data in {**input_tensors, **output_tensors}.items():
            if self.binding_dtype(name) != data.dtype:
                raise TypeError(
                    f'Binding ("{name}") dtype ({self.binding_dtype(name)}) and '
                    f'binding tensor dtype ({data.dtype}) must be the same'
                )

            index = self.engine[name]
            bindings[index] = int(data.data_ptr())

        return bindings, output_tensors

    def run(
        self,
        input_tensors: Dict[str, torch.Tensor],
        output_tensors_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs model and returns result as dictionary, where keys are names of outputs
        and values are corresponding output data as `torch.Tensor`s.

        Parameters
        ----------
        input_tensors : Dict[str, torch.Tensor]
            Dictionary of inputs, where keys are names of inputs and values are
            corresponding input data as `torch.Tensor`s.

        output_tensors_cache : Optional[Dict[str, torch.Tensor]]
            Dictionary of preallocated tensors for model output. Model checks every
            output tensor and resizes one in-place if it is required. You can skip this
            parameter, then output will be allocated by the model.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model output as dictionary, where keys are names of outputs and values are
            corresponding output data as `torch.Tensor`s.

        """
        bindings, output_tensors = self._create_bindings(
            input_tensors=input_tensors,
            output_tensors_cache=output_tensors_cache,
        )
        if not self.context.execute_v2(bindings=bindings):
            raise RuntimeError('execute_v2 FAILED')

        return output_tensors
