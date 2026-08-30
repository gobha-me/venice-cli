#!/usr/bin/env python3
"""Deterministic Venice OpenAPI inventory and drift checker.

The committed lock is JSON so ordinary tests need only the standard library.
PyYAML is imported lazily only by the networked ``refresh`` and ``check-live``
commands; see ``scripts/openapi-requirements.txt``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "https://docs.venice.ai/swagger.yaml"
DEFAULT_LOCK = ROOT / "contracts" / "openapi" / "venice.lock.json"
DEFAULT_MANIFEST = ROOT / "contracts" / "openapi" / "implementation.json"
MAX_SPEC_BYTES = 8 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 30
HTTP_METHODS = ("get", "post", "put", "patch", "delete", "options", "head")
SCHEMA_KEYS = frozenset(
    {
        "additionalProperties",
        "anyOf",
        "deprecated",
        "discriminator",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "nullable",
        "oneOf",
        "pattern",
        "properties",
        "required",
        "type",
    }
)


class ContractError(ValueError):
    """The source, lock, or implementation manifest is not trustworthy."""


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {path}: {exc}") from None
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _load_yaml(raw: bytes) -> dict:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        raise ContractError(
            "PyYAML is required for live OpenAPI work; install "
            "scripts/openapi-requirements.txt"
        ) from None
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ContractError(f"OpenAPI YAML is invalid: {exc}") from None
    if not isinstance(value, dict):
        raise ContractError("OpenAPI document must be an object")
    return value


def fetch_spec(source: str = DEFAULT_SOURCE) -> bytes:
    """Fetch the public spec from the one authorized HTTPS origin, bounded."""
    if source != DEFAULT_SOURCE:
        raise ContractError(f"refusing unapproved OpenAPI source: {source!r}")
    request = urllib.request.Request(
        source,
        headers={
            "Accept": "application/yaml, text/yaml, text/plain",
            "User-Agent": "venice-cli-openapi-contract/1",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
            final_url = response.geturl()
            if final_url != DEFAULT_SOURCE:
                raise ContractError(f"OpenAPI source redirected to {final_url!r}")
            declared = response.headers.get("Content-Length")
            if declared:
                try:
                    if int(declared) > MAX_SPEC_BYTES:
                        raise ContractError("OpenAPI source exceeds the 8 MiB limit")
                except ValueError:
                    raise ContractError("OpenAPI source has invalid Content-Length") from None
            raw = response.read(MAX_SPEC_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ContractError(f"cannot fetch official OpenAPI source: {exc}") from None
    if len(raw) > MAX_SPEC_BYTES:
        raise ContractError("OpenAPI source exceeds the 8 MiB limit")
    if not raw:
        raise ContractError("OpenAPI source is empty")
    return raw


def _json_pointer(root: Mapping[str, Any], pointer: str) -> Any:
    if not pointer.startswith("#/"):
        raise ContractError(f"only local OpenAPI references are supported: {pointer!r}")
    value: Any = root
    for raw_part in pointer[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or part not in value:
            raise ContractError(f"unresolved OpenAPI reference: {pointer!r}")
        value = value[part]
    return value


def _merge_all_of(parts: Sequence[dict]) -> dict:
    merged: dict = {}
    properties: dict = {}
    required = set()
    for part in parts:
        for key, value in part.items():
            if key == "properties":
                if not isinstance(value, dict):
                    raise ContractError("allOf properties must be an object")
                for name, schema in value.items():
                    if name in properties and properties[name] != schema:
                        if not isinstance(properties[name], dict) or not isinstance(schema, dict):
                            raise ContractError(f"conflicting allOf property {name!r}")
                        combined = dict(properties[name])
                        combined.update(schema)
                        properties[name] = combined
                    else:
                        properties[name] = schema
            elif key == "required":
                if not isinstance(value, list):
                    raise ContractError("allOf required must be an array")
                required.update(value)
            elif key in merged and merged[key] != value:
                raise ContractError(f"conflicting allOf keyword {key!r}")
            else:
                merged[key] = value
    if properties:
        merged["properties"] = properties
    if required:
        merged["required"] = sorted(required)
    return merged


def normalize_schema(schema: object, root: Mapping[str, Any], stack: tuple = ()) -> dict:
    if not isinstance(schema, Mapping):
        raise ContractError("request schema must be an object")
    current = dict(schema)
    if "$ref" in current:
        pointer = current.pop("$ref")
        if not isinstance(pointer, str):
            raise ContractError("OpenAPI $ref must be a string")
        if pointer in stack:
            raise ContractError(f"recursive request schema reference: {pointer!r}")
        target = normalize_schema(_json_pointer(root, pointer), root, stack + (pointer,))
        for key, value in current.items():
            if key in target and target[key] != value:
                raise ContractError(f"conflicting sibling beside $ref: {key!r}")
            target[key] = value
        current = target
    if "allOf" in current:
        raw_parts = current.pop("allOf")
        if not isinstance(raw_parts, list) or not raw_parts:
            raise ContractError("allOf must be a non-empty array")
        parts = [normalize_schema(part, root, stack) for part in raw_parts]
        parts.append(current)
        current = _merge_all_of(parts)

    normalized: dict = {}
    for key in sorted(SCHEMA_KEYS & current.keys()):
        value = current[key]
        if key == "properties":
            if not isinstance(value, Mapping):
                raise ContractError("schema properties must be an object")
            normalized[key] = {
                name: normalize_schema(child, root, stack)
                for name, child in sorted(value.items())
            }
        elif key == "items":
            normalized[key] = normalize_schema(value, root, stack)
        elif key in ("anyOf", "oneOf"):
            if not isinstance(value, list) or not value:
                raise ContractError(f"{key} must be a non-empty array")
            normalized[key] = [normalize_schema(part, root, stack) for part in value]
        elif key == "required":
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise ContractError("schema required must be an array of strings")
            normalized[key] = sorted(set(value))
        elif key == "enum":
            if not isinstance(value, list):
                raise ContractError("schema enum must be an array")
            normalized[key] = sorted(value, key=lambda item: json.dumps(item, sort_keys=True))
        elif key == "additionalProperties" and isinstance(value, Mapping):
            normalized[key] = normalize_schema(value, root, stack)
        else:
            normalized[key] = deepcopy(value)
    return normalized


def _operation_key(method: str, path: str) -> str:
    return f"{method.upper()} {path}"


def _operation_inventory(spec: Mapping[str, Any]) -> list:
    paths = spec.get("paths")
    if not isinstance(paths, Mapping):
        raise ContractError("OpenAPI document has no paths object")
    inventory = []
    for path, path_item in sorted(paths.items()):
        if not isinstance(path_item, Mapping):
            raise ContractError(f"path item {path!r} must be an object")
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if operation is None:
                continue
            if not isinstance(operation, Mapping):
                raise ContractError(f"operation {_operation_key(method, path)} is invalid")
            inventory.append(
                {
                    "key": _operation_key(method, path),
                    "operation_id": operation.get("operationId"),
                }
            )
    return inventory


def _extract_operation(spec: Mapping[str, Any], key: str) -> dict:
    try:
        method, path = key.split(" ", 1)
    except ValueError:
        raise ContractError(f"invalid operation key {key!r}") from None
    paths = spec.get("paths")
    path_item = paths.get(path) if isinstance(paths, Mapping) else None
    operation = path_item.get(method.lower()) if isinstance(path_item, Mapping) else None
    if not isinstance(operation, Mapping):
        raise ContractError(f"implemented operation missing upstream: {key}")

    parameters = []
    raw_parameters = []
    if isinstance(path_item.get("parameters"), list):
        raw_parameters.extend(path_item["parameters"])
    if isinstance(operation.get("parameters"), list):
        raw_parameters.extend(operation["parameters"])
    for parameter in raw_parameters:
        if isinstance(parameter, Mapping) and "$ref" in parameter:
            parameter = _json_pointer(spec, parameter["$ref"])
        if not isinstance(parameter, Mapping):
            raise ContractError(f"invalid parameter in {key}")
        name = parameter.get("name")
        location = parameter.get("in")
        if not isinstance(name, str) or not isinstance(location, str):
            raise ContractError(f"unnamed parameter in {key}")
        parameters.append(
            {
                "in": location,
                "name": name,
                "required": bool(parameter.get("required", False)),
                "schema": normalize_schema(parameter.get("schema", {}), spec),
            }
        )
    parameters.sort(key=lambda item: (item["in"], item["name"]))

    bodies = {}
    request_body = operation.get("requestBody")
    if isinstance(request_body, Mapping) and "$ref" in request_body:
        request_body = _json_pointer(spec, request_body["$ref"])
    if request_body is not None:
        if not isinstance(request_body, Mapping):
            raise ContractError(f"invalid requestBody in {key}")
        content = request_body.get("content", {})
        if not isinstance(content, Mapping):
            raise ContractError(f"invalid requestBody content in {key}")
        for media_type, media in sorted(content.items()):
            if not isinstance(media, Mapping) or "schema" not in media:
                raise ContractError(f"missing {media_type} schema in {key}")
            bodies[media_type] = normalize_schema(media["schema"], spec)

    return {
        "operation_id": operation.get("operationId"),
        "parameters": parameters,
        "request_bodies": bodies,
    }


def build_lock(
    raw: bytes, manifest: Mapping[str, Any], *, allow_missing: bool = False
) -> dict:
    spec = _load_yaml(raw)
    info = spec.get("info")
    version = info.get("version") if isinstance(info, Mapping) else None
    if not isinstance(version, str) or not version:
        raise ContractError("OpenAPI document has no info.version")
    operations = manifest.get("operations")
    if not isinstance(operations, Mapping) or not operations:
        raise ContractError("implementation manifest has no operations")
    implemented = {}
    for key in sorted(operations):
        try:
            implemented[key] = _extract_operation(spec, key)
        except ContractError as exc:
            if allow_missing and str(exc).startswith("implemented operation missing upstream:"):
                continue
            raise
    return {
        "format": 1,
        "source": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "url": DEFAULT_SOURCE,
            "version": version,
        },
        "all_operations": _operation_inventory(spec),
        "implemented_operations": implemented,
    }


def _top_level_fields(schema: Mapping[str, Any]) -> set:
    fields = set(schema.get("properties", {}))
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword, [])
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, Mapping):
                    fields.update(_top_level_fields(branch))
    return fields


def _top_level_required(schema: Mapping[str, Any]) -> set:
    required = set(schema.get("required", []))
    for keyword in ("anyOf", "oneOf"):
        branches = schema.get(keyword, [])
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, Mapping):
                    required.update(_top_level_required(branch))
    return required


def _validate_classification(
    *, actual: set, implemented: object, omissions: object, label: str
) -> list:
    errors = []
    if not isinstance(implemented, list) or any(not isinstance(v, str) for v in implemented):
        return [f"{label}: implemented must be an array of strings"]
    if not isinstance(omissions, Mapping):
        return [f"{label}: intentional_omissions must be an object"]
    implemented_set = set(implemented)
    overlap = implemented_set & set(omissions)
    if overlap:
        errors.append(f"{label}: fields both implemented and omitted: {sorted(overlap)}")
    classified = implemented_set | set(omissions)
    if classified != actual:
        missing = sorted(actual - classified)
        extra = sorted(classified - actual)
        if missing:
            errors.append(f"{label}: unclassified upstream fields: {missing}")
        if extra:
            errors.append(f"{label}: fields absent upstream: {extra}")
    for field, reason in omissions.items():
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{label}: omission {field!r} needs a reason")
    return errors


def validate_manifest(lock: Mapping[str, Any], manifest: Mapping[str, Any]) -> list:
    errors = []
    locked = lock.get("implemented_operations")
    declared = manifest.get("operations")
    if not isinstance(locked, Mapping) or not isinstance(declared, Mapping):
        return ["lock and manifest must contain implemented operations"]
    if set(locked) != set(declared):
        errors.append("lock and manifest operation sets differ; run the refresh command")
    for key in sorted(set(locked) & set(declared)):
        contract = locked[key]
        implementation = declared[key]
        if not isinstance(contract, Mapping) or not isinstance(implementation, Mapping):
            errors.append(f"{key}: operation entries must be objects")
            continue
        consumers = implementation.get("consumers")
        if not isinstance(consumers, list) or not consumers or any(
            not isinstance(value, str) or ":" not in value for value in consumers
        ):
            errors.append(f"{key}: consumers must be non-empty surface:name strings")

        actual_parameters = {
            f"{item['in']}:{item['name']}" for item in contract.get("parameters", [])
        }
        required_parameters = {
            f"{item['in']}:{item['name']}"
            for item in contract.get("parameters", [])
            if item.get("required")
        }
        parameters = implementation.get("parameters", {})
        if not isinstance(parameters, Mapping):
            errors.append(f"{key}: parameters must be an object")
        else:
            errors.extend(
                _validate_classification(
                    actual=actual_parameters,
                    implemented=parameters.get("implemented", []),
                    omissions=parameters.get("intentional_omissions", {}),
                    label=f"{key} parameters",
                )
            )
            omitted_parameters = parameters.get("intentional_omissions", {})
            if isinstance(omitted_parameters, Mapping):
                omitted_required = required_parameters & set(omitted_parameters)
                if omitted_required:
                    errors.append(
                        f"{key} parameters: required parameters cannot be omitted: "
                        f"{sorted(omitted_required)}"
                    )

        actual_bodies = contract.get("request_bodies", {})
        declared_bodies = implementation.get("request_bodies", {})
        if not isinstance(actual_bodies, Mapping) or not isinstance(declared_bodies, Mapping):
            errors.append(f"{key}: request_bodies must be objects")
            continue
        if set(actual_bodies) != set(declared_bodies):
            errors.append(
                f"{key}: request media types differ: upstream={sorted(actual_bodies)} "
                f"manifest={sorted(declared_bodies)}"
            )
        for media_type in sorted(set(actual_bodies) & set(declared_bodies)):
            declared_body = declared_bodies[media_type]
            if not isinstance(declared_body, Mapping):
                errors.append(f"{key} {media_type}: declaration must be an object")
                continue
            unsupported = declared_body.get("unsupported")
            if unsupported is not None:
                if not isinstance(unsupported, str) or not unsupported.strip():
                    errors.append(f"{key} {media_type}: unsupported needs a reason")
                if set(declared_body) != {"unsupported"}:
                    errors.append(f"{key} {media_type}: unsupported media cannot classify fields")
                continue
            fields = _top_level_fields(actual_bodies[media_type])
            errors.extend(
                _validate_classification(
                    actual=fields,
                    implemented=declared_body.get("implemented", []),
                    omissions=declared_body.get("intentional_omissions", {}),
                    label=f"{key} {media_type}",
                )
            )
            omissions = declared_body.get("intentional_omissions", {})
            if isinstance(omissions, Mapping):
                omitted_required = _top_level_required(actual_bodies[media_type]) & set(
                    omissions
                )
                if omitted_required:
                    errors.append(
                        f"{key} {media_type}: required fields cannot be omitted: "
                        f"{sorted(omitted_required)}"
                    )

        constraints = implementation.get("model_catalog_constraints", {})
        if not isinstance(constraints, Mapping):
            errors.append(f"{key}: model_catalog_constraints must be an object")
        else:
            body_fields = set()
            for body in actual_bodies.values():
                if isinstance(body, Mapping):
                    body_fields.update(_top_level_fields(body))
            for field, pointer in constraints.items():
                if field not in body_fields:
                    errors.append(f"{key}: catalog constraint field absent upstream: {field}")
                if not isinstance(pointer, str) or not pointer.startswith("/models:"):
                    errors.append(f"{key}: invalid catalog constraint for {field!r}")
    return errors


def _operation_keys(lock: Mapping[str, Any]) -> set:
    inventory = lock.get("all_operations", [])
    return {
        item["key"]
        for item in inventory
        if isinstance(item, Mapping) and isinstance(item.get("key"), str)
    }


def _diff(old: object, new: object, path: str = "") -> list:
    """Return actionable (severity, path, old, new) tuples."""
    changes = []
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        old_keys, new_keys = set(old), set(new)
        for key in sorted(old_keys - new_keys):
            severity = "breaking"
            if key in ("required",):
                severity = "review"
            changes.append((severity, f"{path}/{key}", old[key], None))
        for key in sorted(new_keys - old_keys):
            severity = "review"
            if key == "required":
                severity = "breaking"
            changes.append((severity, f"{path}/{key}", None, new[key]))
        for key in sorted(old_keys & new_keys):
            changes.extend(_diff(old[key], new[key], f"{path}/{key}"))
        return changes
    if isinstance(old, list) and isinstance(new, list):
        if path.endswith("/required"):
            old_set, new_set = set(old), set(new)
            for value in sorted(new_set - old_set):
                changes.append(("breaking", path, None, value))
            for value in sorted(old_set - new_set):
                changes.append(("review", path, value, None))
            return changes
        if path.endswith("/enum"):
            old_values = {json.dumps(v, sort_keys=True): v for v in old}
            new_values = {json.dumps(v, sort_keys=True): v for v in new}
            for value in sorted(old_values.keys() - new_values.keys()):
                changes.append(("breaking", path, old_values[value], None))
            for value in sorted(new_values.keys() - old_values.keys()):
                changes.append(("review", path, None, new_values[value]))
            return changes
        if len(old) == len(new):
            for index, (old_item, new_item) in enumerate(zip(old, new)):
                changes.extend(_diff(old_item, new_item, f"{path}/{index}"))
        elif old != new:
            changes.append(("review", path, old, new))
        return changes
    if old == new:
        return changes
    severity = "breaking"
    key = path.rsplit("/", 1)[-1]
    if key in ("minimum", "exclusiveMinimum", "minLength", "minItems"):
        try:
            severity = "breaking" if new > old else "review"
        except TypeError:
            severity = "breaking"
    elif key in ("maximum", "exclusiveMaximum", "maxLength", "maxItems"):
        try:
            severity = "breaking" if new < old else "review"
        except TypeError:
            severity = "breaking"
    elif key in ("deprecated",):
        severity = "review"
    changes.append((severity, path or "/", old, new))
    return changes


def compare_locks(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict:
    old_all, new_all = _operation_keys(old), _operation_keys(new)
    old_impl = old.get("implemented_operations", {})
    new_impl = new.get("implemented_operations", {})
    changes = []
    if isinstance(old_impl, Mapping) and isinstance(new_impl, Mapping):
        for key in sorted(set(old_impl) | set(new_impl)):
            if key not in new_impl:
                changes.append(("breaking", f"/{key}", old_impl.get(key), None))
            elif key not in old_impl:
                changes.append(("review", f"/{key}", None, new_impl.get(key)))
            else:
                changes.extend(_diff(old_impl[key], new_impl[key], f"/{key}"))
    return {
        "implemented_changes": changes,
        "unsupported_added": sorted((new_all - old_all) - set(new_impl)),
        "unsupported_removed": sorted((old_all - new_all) - set(old_impl)),
        "source_changed": old.get("source") != new.get("source"),
    }


def _brief(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return rendered if len(rendered) <= 160 else rendered[:157] + "..."


def render_report(report: Mapping[str, Any], old: Mapping[str, Any], new: Mapping[str, Any]) -> str:
    old_source = old.get("source", {})
    new_source = new.get("source", {})
    lines = [
        "# Venice OpenAPI drift report",
        "",
        f"Baseline: version `{old_source.get('version')}`, SHA-256 `{old_source.get('sha256')}`",
        f"Live: version `{new_source.get('version')}`, SHA-256 `{new_source.get('sha256')}`",
        "",
    ]
    changes = report.get("implemented_changes", [])
    if changes:
        lines.extend(["## Implemented-operation changes", ""])
        for severity, path, before, after in changes:
            lines.append(
                f"- **{severity}** `{path}`: `{_brief(before)}` -> `{_brief(after)}`"
            )
        lines.append("")
    else:
        lines.extend(["No implemented-operation drift detected.", ""])
    for heading, key in (
        ("New unsupported operations (informational)", "unsupported_added"),
        ("Removed unsupported operations (informational)", "unsupported_removed"),
    ):
        values = report.get(key, [])
        if values:
            lines.extend([f"## {heading}", ""])
            lines.extend(f"- `{value}`" for value in values)
            lines.append("")
    if report.get("source_changed") and not changes:
        lines.append("The upstream document changed, but normalized implemented contracts did not.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _append_github_summary(text: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(text)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "check-live", "refresh"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--github-summary", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = _read_json(args.manifest)
        if args.command == "check":
            lock = _read_json(args.lock)
            errors = validate_manifest(lock, manifest)
            if errors:
                for error in errors:
                    print(f"openapi-contract: {error}", file=sys.stderr)
                return 1
            print(
                "openapi-contract: offline inventory OK "
                f"({len(lock['implemented_operations'])} implemented operations)"
            )
            return 0

        raw = fetch_spec(args.source)
        if args.command == "refresh":
            live = build_lock(raw, manifest)
            errors = validate_manifest(live, manifest)
            if errors:
                for error in errors:
                    print(f"openapi-contract: {error}", file=sys.stderr)
                return 1
            _write_json(args.lock, live)
            print(
                f"openapi-contract: wrote {args.lock} from version "
                f"{live['source']['version']} ({live['source']['sha256']})"
            )
            return 0

        baseline = _read_json(args.lock)
        errors = validate_manifest(baseline, manifest)
        if errors:
            for error in errors:
                print(f"openapi-contract: {error}", file=sys.stderr)
            return 1
        live = build_lock(raw, manifest, allow_missing=True)
        report = compare_locks(baseline, live)
        rendered = render_report(report, baseline, live)
        sys.stdout.write(rendered)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered, encoding="utf-8")
        if args.github_summary:
            _append_github_summary(rendered)
        return 1 if report["implemented_changes"] else 0
    except ContractError as exc:
        print(f"openapi-contract: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
