import ast
import builtins
from pathlib import Path

import nbformat


notebook = nbformat.read(
    Path("notebooks/04_new_models.ipynb"),
    as_version=4,
)

defined = set(dir(builtins))
possibly_used_before_defined = []

for cell_index, cell in enumerate(notebook.cells):
    if cell.cell_type != "code":
        continue

    tree = ast.parse(cell.source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for element in target.elts:
                        if isinstance(element, ast.Name):
                            defined.add(element.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
        elif isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.optional_vars, ast.Name):
                    defined.add(item.optional_vars.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in defined:
                possibly_used_before_defined.append(
                    (cell_index, node.id)
                )

ignored = {
    "globals",
    "range",
    "len",
    "list",
    "sorted",
    "dict",
    "zip",
    "next",
    "str",
    "print",
    "ImportError",
    "ValueError",
    "FileNotFoundError",
    "RuntimeError",
}

result = [
    item
    for item in possibly_used_before_defined
    if item[1] not in ignored
]

print(sorted(set(result)))
