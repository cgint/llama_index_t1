import ast

# Define the visitor class
class MethodCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.methods = {}
        self.current_method = None

    def visit_FunctionDef(self, node):
        # Record the function name
        self.current_method = node.name
        self.methods[self.current_method] = []
        self.generic_visit(node)
        self.current_method = None

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            method_name = node.func.id
        else:
            method_name = "unknown"

        if self.current_method:
            self.methods[self.current_method].append(method_name)
        self.generic_visit(node)

# Load the user's Python file
file_paths = ['llamaindex_simple_graph_rag.py', 'lib/engine_query.py', 'lib/graph.py', 'lib/__init__.py', 'lib/llm.py', 'lib/loader.py', 'lib/question_store.py', 'lib/rag_process.py', 'lib/vector.py']
code = ""

for file_path in file_paths:
    with open(file_path, 'r') as file:
        code += file.read()
        code += "\n\n"

# Parse the file
parsed_code = ast.parse(code)

# Visit the nodes
visitor = MethodCallVisitor()
visitor.visit(parsed_code)

# Print the methods and their calls
for method, calls in visitor.methods.items():
    print(f"Method: {method}")
    print("Calls:")
    for call in calls:
        print(f"  - {call}")

