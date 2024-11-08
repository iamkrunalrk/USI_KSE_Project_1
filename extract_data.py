import os
import pandas as pd
import ast

# List to store extracted data
data_list = []

class KseVisitor(ast.NodeVisitor):
    """
    Custom AST NodeVisitor to traverse through Python files
    and extract function, class, and method details.
    """

    def __init__(self):
        # Initialize path to empty string
        self.path = ""

    def is_blacklist(self, node):
        """
        Check if a node (function or class) should be ignored.
        A node is ignored if its name starts with '_', is 'main', or contains 'test'.
        """
        name = node.name
        return name.startswith("_") or name == "main" or "test" in name.lower()

    def visit_path(self, path):
        """
        Parse and visit all AST nodes in a given Python file.
        """
        self.path = path
        # Read file content, parse it and visit its AST nodes
        with open(path, "r") as file:
            self.visit(ast.parse(file.read()))

    def add_data(self, node, node_type):
        """
        Extract relevant details (name, path, line, type, comment) from the node
        and add them to the data list.
        """
        name = node.name
        line = node.lineno
        path = self.path
        # Get the docstring for the node (function or class)
        docstring = ast.get_docstring(node)
        # If no docstring exists, use a blank space, otherwise clean it up
        comment = " " if docstring is None else " ".join(docstring.splitlines()).strip()

        # Append the extracted data as a tuple
        data_list.append([name, path, line, node_type, comment])

    def visit_FunctionDef(self, node):
        """
        Visit a function definition node. If it's not blacklisted, add its data.
        """
        if not self.is_blacklist(node):
            self.add_data(node, "Function")

    def visit_ClassDef(self, node):
        """
        Visit a class definition node. If it's not blacklisted, add its data.
        Also, visit methods inside the class.
        """
        if not self.is_blacklist(node):
            self.add_data(node, "Class")
            # Traverse methods inside the class
            for child_node in node.body:
                if isinstance(child_node, ast.FunctionDef) and not self.is_blacklist(child_node):
                    self.add_data(child_node, "Method")

# List to store all Python file paths
paths_list = []
# Directory to start the search
root_path = "tensorflow/"

# Walk through the directory and collect Python file paths
for root, dirs, files in os.walk(root_path):
    for file in files:
        if file.endswith(".py"):
            paths_list.append(os.path.join(root, file))

print(f"Number of Python files: {len(paths_list)}")

# Create an instance of the visitor
visitor = KseVisitor()

# Visit each Python file in the paths list
for file_path in paths_list:
    visitor.visit_path(file_path)

# Convert the collected data into a pandas DataFrame
df = pd.DataFrame(data_list, columns=["Name", "Path", "Line", "Type", "Comment"])

# Save the data to a CSV file
df.to_csv("data.csv", index=False)

# Print counts of classes, functions, and methods
print(f"Classes: {df['Type'].value_counts().get('Class', 0)}")
print(f"Functions: {df['Type'].value_counts().get('Function', 0)}")
print(f"Methods: {df['Type'].value_counts().get('Method', 0)}")
