import torch
from typing import Optional, Any
from collections import OrderedDict
from functools import cached_property


class TensorDict(dict):
    def __init__(self, base_dict: dict = None, fields: Optional[dict] = None) -> None:
        """A dictionary of tensors with types for each field.

        Args:
            base_dict (dict, optional): Starting dictionary with field names as keys
                and tensors as values. Defaults to empty dict.
            fields (Optional[dict], optional): Dictionary with keys: "numerical" and "categorical"
                and values: list of field names corresponding to the type given by the key.
                Defaults to empty lists.
        """
        super().__init__(base_dict if base_dict is not None else {})
        if fields is None:
            fields = {
                "numerical": [],
                "categorical": [],
            }
        self.fields = fields
        self.numerical = fields["numerical"]
        self.categorical = fields["categorical"]

    @property
    def subset_numerical(self):
        return {field: self[field] for field in self.subset_numerical}

    @property
    def subset_categorical(self):
        return {field: self[field] for field in self.subset_categorical}

    @property
    def subset_text(self):
        return {field: self[field] for field in self.subset_text}

    def to(self, device):
        for field in self:
            self[field] = self[field].to(device)
        return self

    @property
    def iloc(self):
        return iloc(self)

    def __invert__(self):
        return TensorDict(
            {field: ~tensor for field, tensor in self.items()},
            fields=self.fields,
        )

    def to_tensor(self):
        return torch.hstack(list(self.values()))

    def float(self):
        return TensorDict(
            {field: tensor.float() for field, tensor in self.items()},
            fields=self.fields,
        )

    def float_(self):
        for field in self:
            self[field] = self[field].float()
        return self

    def bool(self):
        return TensorDict(
            {field: tensor.bool() for field, tensor in self.items()},
            fields=self.fields,
        )

    def bool_(self):
        for field in self:
            self[field] = self[field].bool()
        return self

    def detach(self):
        return TensorDict(
            {field: tensor.detach() for field, tensor in self.items()},
            fields=self.fields,
        )

    def detach_(self):
        for field in self:
            self[field] = self[field].detach()
        return self

    def copy(self):
        return TensorDict(
            {field: tensor.clone() for field, tensor in self.items()},
            fields=self.fields,
        )

    def size(self, column=None):
        size = None
        if column is not None:
            return len(self[column])
        for tensor in self.values():
            if size is None:
                size = len(tensor)
            else:
                assert size == len(tensor), "All tensors must have the same length"
        return size

    def device(self, idx=None, column=None):
        if idx is not None:
            return self.values()[idx].device
        if column is not None:
            return self[column].device
        device = None
        for tensor in self.values():
            if device is None:
                device = tensor.device
            else:
                assert device == tensor.device, "All tensors must be on the same device"
        return device

    def __getitem__(self, __key: Any) -> Any:
        if isinstance(__key, list):
            return TensorDict(
                {key: self[key] for key in __key},
                fields=self.fields,
            )
        return super().__getitem__(__key)


class iloc:
    def __init__(self, data: TensorDict):
        self.data = data

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            key = idx[0]
            indices = idx[1]
            return self.data[key][indices]
        else:
            return TensorDict(
                {key: self.data[key][idx] for key in self.data},
                fields=self.data.fields,
            )

    def __call__(self, idx):
        return self[idx]


class Fields(OrderedDict):
    @cached_property
    def all_fields(self):
        all_fields = []
        for field_values in self.values():
            all_fields.extend(field_values)
        return all_fields

    def type(self, field):
        for field_type, fields in self.items():
            if field in fields:
                return field_type
        raise ValueError(f"{field} not found in any field type")
