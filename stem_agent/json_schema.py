from __future__ import annotations

from typing import Any


class SchemaValidationError(ValueError):
    pass


def check_schema(schema: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    _check_schema(schema, path, errors)
    return errors


def validate_instance(value: Any, schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _validate_instance(value, schema, "$", errors)
    return errors


def _check_schema(schema: Any, path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict):
        errors.append(f"{path}: schema must be an object")
        return

    schema_type = schema.get("type")
    if schema_type is not None:
        allowed = {
            "object",
            "array",
            "string",
            "integer",
            "number",
            "boolean",
            "null",
        }
        types = schema_type if isinstance(schema_type, list) else [schema_type]
        if not types or any(not isinstance(item, str) or item not in allowed for item in types):
            errors.append(f"{path}.type: unsupported JSON Schema type")

    required = schema.get("required")
    if required is not None and (
        not isinstance(required, list)
        or any(not isinstance(item, str) for item in required)
    ):
        errors.append(f"{path}.required: must be a list of strings")

    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            errors.append(f"{path}.properties: must be an object")
        else:
            for name, child in properties.items():
                if not isinstance(name, str):
                    errors.append(f"{path}.properties: property names must be strings")
                    continue
                _check_schema(child, f"{path}.properties.{name}", errors)

    items = schema.get("items")
    if items is not None:
        _check_schema(items, f"{path}.items", errors)

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        _check_schema(additional, f"{path}.additionalProperties", errors)
    elif additional is not None and not isinstance(additional, bool):
        errors.append(f"{path}.additionalProperties: must be boolean or schema")

    enum = schema.get("enum")
    if enum is not None and not isinstance(enum, list):
        errors.append(f"{path}.enum: must be a list")

    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if variants is not None:
            if not isinstance(variants, list) or not variants:
                errors.append(f"{path}.{keyword}: must be a non-empty list")
            else:
                for index, child in enumerate(variants):
                    _check_schema(child, f"{path}.{keyword}[{index}]", errors)


def _validate_instance(
    value: Any,
    schema: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}")
        return

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: expected one of {schema['enum']!r}")
        return

    if "anyOf" in schema:
        variants = schema["anyOf"]
        if not any(not validate_instance(value, variant) for variant in variants):
            errors.append(f"{path}: did not match anyOf")
        return

    if "oneOf" in schema:
        matches = sum(1 for variant in schema["oneOf"] if not validate_instance(value, variant))
        if matches != 1:
            errors.append(f"{path}: matched {matches} oneOf variants")
        return

    schema_type = schema.get("type")
    if schema_type is not None and not _matches_type(value, schema_type):
        errors.append(f"{path}: expected type {schema_type!r}")
        return

    if isinstance(value, dict):
        _validate_object(value, schema, path, errors)
    elif isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate_instance(item, items, f"{path}[{index}]", errors)


def _validate_object(
    value: dict[str, Any],
    schema: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    for name in schema.get("required", []):
        if name not in value:
            errors.append(f"{path}.{name}: missing required property")

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for name, child in properties.items():
            if name in value and isinstance(child, dict):
                _validate_instance(value[name], child, f"{path}.{name}", errors)

    additional = schema.get("additionalProperties", True)
    if additional is False:
        for name in value:
            if name not in properties:
                errors.append(f"{path}.{name}: additional property is not allowed")
    elif isinstance(additional, dict):
        for name, child in value.items():
            if name not in properties:
                _validate_instance(child, additional, f"{path}.{name}", errors)


def _matches_type(value: Any, schema_type: str | list[str]) -> bool:
    if isinstance(schema_type, list):
        return any(_matches_type(value, item) for item in schema_type)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "null":
        return value is None
    return False
