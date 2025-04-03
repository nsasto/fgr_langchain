from itertools import chain
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic._internal._model_construction import ModelMetaclass

# LLM Models
####################################################################################################
class _BaseModelAliasMeta(ModelMetaclass):
    """Metaclass for `BaseModelAlias`."""

    def __getattr__(cls, name: str) -> Any:
        """Get attribute from the inner `Model` class.

        Args:
            name: The name of the attribute to get.

        Returns:
            The attribute from the inner `Model` class.

        Raises:
            AttributeError: If the attribute is not found.
        """
        return super().__getattribute__("Model").__getattribute__(name)    

    
class BaseModelAlias:
    """Base class for data classes that use Pydantic models with aliases.

    This class provides a way to define a Pydantic model with an alias and
    automatically map between the data class and the Pydantic model.
    """

    class Model(BaseModel):
        """
        Inner Pydantic model class.

        Attributes:

        This class should be subclassed to define the actual Pydantic model.
        """
        @staticmethod
        def to_dataclass(pydantic: Any) -> Any:
            raise NotImplementedError

    def to_str(self) -> str:
        raise NotImplementedError

    __pydantic_model__: BaseModel
    """Return the inner Pydantic model instance.

    Returns: The inner Pydantic model instance.
    """


####################################################################################################
# LLM Dumping to strings
####################################################################################################


def dump_to_csv(
    data: Iterable[object],
    fields: List[str],
    separator: str = "\t",
    with_header: bool = False,
    **values: Dict[str, List[Any]],
) -> List[str]:
    """Dump a list of objects to a CSV-formatted string.

    Args:
        data: An iterable of objects to dump.
        fields: A list of field names to include in the CSV.
        separator: The separator character to use between fields.
        with_header: Whether to include a header row with field names. Default is False.
        **values: Additional key-value pairs to include as columns in the CSV.

    Returns:
        A list of strings, each representing a row in the CSV.
        If `with_header` is True, the first row will be a header row.

    """
    rows = list(
        chain(
            (separator.join(chain(fields, values.keys())),) if with_header else (),
            chain(
                separator.join(
                    chain(
                        (str(getattr(d, field)).replace("\n", "  ").replace("\t", " ") for field in fields),
                        (str(v).replace("\n", "  ").replace("\t", " ") for v in vs),
                    )
                )
                for d, *vs in zip(data, *values.values())
            ),
        )
    )
    return rows


def dump_to_reference_list(data: Iterable[object], separator: str = "\n=====\n\n"):
    """Dump a list of objects to a reference list-formatted string.

    Args:
        data: An iterable of objects to dump.
        separator: The separator string to use between items.

    Returns:
        A list of strings, each representing an item in the reference list.        
    """
    return [f"[{i + 1}]  {d}{separator}" for i, d in enumerate(data)]  # type: ignore[union-attr]


####################################################################################################
# Response Models
####################################################################################################


class TAnswer(BaseModel):
    """Pydantic model for an answer.

    Attributes:
        answer: The answer string.
    """
    answer: str    


class TEditRelation(BaseModel):
    ids: List[int] = Field(..., description="Ids of the facts that you are combining into one")
    description: str = Field(
        ..., description="Summarized description of the combined facts, in detail and comprehensive"
    )


class TEditRelationList(BaseModel):
    groups: List[TEditRelation] = Field(
        ...,
        description="List of new fact groups; include only groups of more than one fact",
        alias="grouped_facts",
    )
    

class TEntityDescription(BaseModel):
    """Pydantic model for an entity description.

    Attributes:
        description: The description of the entity.
    """
    description: str    


class TQueryEntities(BaseModel):
    named: List[str] = Field(
        ...,
        description=("List of named entities extracted from the query"),
    )
    generic: List[str] = Field(
        ...,
        description=("List of generic entities extracted from the query"),
    )    

    @field_validator("named", mode="before")
    @classmethod
    def uppercase_named(cls, value: List[str]):
        return [e.upper() for e in value] if value else value

    # @field_validator("generic", mode="before")
    # @classmethod
    # def uppercase_generic(cls, value: List[str]):
    #     return [e.upper() for e in value] if value else value
