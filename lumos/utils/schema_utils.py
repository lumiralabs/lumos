from typing import Type, get_origin, get_args, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class SchemaParsingError(ValueError):
    """Error raised when a Pydantic model contains dict type fields."""

    pass


def validate_model_types(model_class: Type[T]) -> None:
    """
    Validates a Pydantic model's type annotations to ensure they conform to allowed types.

    Args:
        model_class: The Pydantic model class to validate

    Raises:
        SchemaParsingError: If the model contains disallowed types like dict. Use Pydantic BaseModel instead.
    """
    for field_name, field_type in model_class.__annotations__.items():
        origin_type = get_origin(field_type)

        # Check if the type is a dict or has dict in its type args
        if origin_type is dict:
            raise SchemaParsingError(
                f"Field '{field_name}' uses dict type which is not allowed. "
                "Use a Pydantic BaseModel class to define the structure instead. Example:\n"
                "class MyModel(BaseModel):\n"
                "    field_name: MyCustomModel"
            )

        # If it's a container type like list, check its type arguments
        if origin_type is not None:
            type_args = get_args(field_type)
            for arg in type_args:
                if get_origin(arg) is dict:
                    raise SchemaParsingError(
                        f"Field '{field_name}' contains nested dict type which is not allowed. "
                        "Use a Pydantic BaseModel class to define the structure instead. Example:\n"
                        "class NestedModel(BaseModel):\n"
                        "    nested_field: MyCustomModel\n"
                        "class ParentModel(BaseModel):\n"
                        "    field_name: list[NestedModel]"
                    )
