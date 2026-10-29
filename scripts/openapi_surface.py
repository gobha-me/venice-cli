"""Inventory every public operation and its request/response schema fields.

This complements the curated request drift gate in openapi_contract.py. Unsupported
operations stay visible; a snapshot is evidence of availability, not a claim that
the CLI or MCP implements them. Only the public Swagger is fetched, without auth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import openapi_contract as contract


DEFAULT_OUTPUT = contract.ROOT / "contracts/openapi/surface.json"


def build_surface(raw: bytes, manifest: dict) -> dict:
    """Resolve request and response schemas for all upstream operations."""
    spec = contract._load_yaml(raw)
    lock = contract.build_lock(raw, manifest)
    operations = {}
    for item in lock["all_operations"]:
        key = item["key"]
        method, path = key.split(" ", 1)
        operation = spec["paths"][path][method.lower()]
        entry = contract._extract_operation(spec, key)
        request = operation.get("requestBody", {})
        if "$ref" in request:
            request = contract._json_pointer(spec, request["$ref"])
        entry["request_body_required"] = bool(request.get("required", False))
        entry["responses"] = {}
        for status, response in operation.get("responses", {}).items():
            if "$ref" in response:
                response = contract._json_pointer(spec, response["$ref"])
            content = {
                media: contract.normalize_schema(body.get("schema", {}), spec)
                for media, body in response.get("content", {}).items()
            }
            headers = {}
            for name, header in response.get("headers", {}).items():
                if "$ref" in header:
                    header = contract._json_pointer(spec, header["$ref"])
                headers[name] = contract.normalize_schema(header.get("schema", {}), spec)
            entry["responses"][str(status)] = {"content": content, "headers": headers}
        entry["coverage"] = manifest["operations"].get(key, {
            "unsupported": "No dedicated client command or MCP tool; generic HTTP transport only."
        })
        operations[key] = entry
    return {"format": 1, "source": lock["source"], "operations": operations}


def render_surface(surface: dict) -> str:
    """List every operation and classify its top-level input fields for review."""
    lines = [
        "# Public API surface inventory", "",
        f"Source: {surface['source']['url']} (version `{surface['source']['version']}`).",
        "",
        "Generated with `scripts/openapi_surface.py`. Full nested request and response",
        "fields, unions, enums, bounds, and response headers are in [surface.json](surface.json).",
        "Coverage describes the union of CLI, internal, agent, and MCP consumers;",
        "it does not claim that every field has a CLI flag or MCP argument.",
        "Unimplemented operations are existing surface gaps, not newly added endpoints.",
        "",
    ]
    for key, operation in surface["operations"].items():
        coverage = operation["coverage"]
        lines.extend([f"## `{key}`", ""])
        if "unsupported" in coverage:
            lines.extend([coverage["unsupported"], ""])
        else:
            lines.extend(["Consumers: " + ", ".join(f"`{c}`" for c in coverage["consumers"]) + ".", ""])
        params = operation["parameters"]
        if params:
            lines.extend(["Parameters: " + ", ".join(
                f"`{p['in']}:{p['name']}`" + (" (required)" if p["required"] else "")
                for p in params
            ) + ".", ""])
        for media, schema in operation["request_bodies"].items():
            declared = coverage.get("request_bodies", {}).get(media, {})
            fields = sorted(contract._top_level_fields(schema))
            lines.extend([f"Request `{media}`:", ""])
            if "unsupported" in declared:
                lines.extend([declared["unsupported"], ""])
            for field in fields:
                if field in declared.get("implemented", []):
                    status = "implemented"
                else:
                    status = declared.get("intentional_omissions", {}).get(field, "unsupported")
                lines.append(f"- `{field}`: {status}")
            if not fields:
                lines.append("No named schema properties published.")
            lines.append("")
        lines.extend(["Responses: " + ", ".join(f"`{s}`" for s in operation["responses"]) + ".", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    """Write reproducible public inventory JSON and its reviewable Markdown index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-file", type=Path, help="Use a previously fetched public Swagger.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = args.spec_file.read_bytes() if args.spec_file else contract.fetch_spec()
    surface = build_surface(raw, contract._read_json(contract.DEFAULT_MANIFEST))
    contract._write_json(args.output, surface)
    args.output.with_suffix(".md").write_text(render_surface(surface), encoding="utf-8")
    print(f"openapi-surface: {len(surface['operations'])} operations recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
