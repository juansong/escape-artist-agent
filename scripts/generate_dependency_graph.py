from graphviz import Digraph
import os
import shutil
import sys

def generate_dependency_graph(output_dir="docs", filename="dependency_graph"):
    """
    Generate a dependency graph of Escape Artist Agent modules using Graphviz.
    Saves the graph as a PNG in the docs/ folder.
    """
    # Check if Graphviz 'dot' executable is available
    if shutil.which("dot") is None:
        print("❌ Graphviz 'dot' binary not found!")
        print("Please install Graphviz system package:")
        print("macOS (Homebrew): brew install graphviz")
        print("Ubuntu/Debian: sudo apt-get install graphviz")
        print("Windows (Chocolatey): choco install graphviz")
        sys.exit(1)

    dot = Digraph(comment="Escape Artist Agent Dependency Graph", format="png")
    dot.attr(rankdir="LR", size="8")

    # Nodes
    dot.node("train_mc", "train_mc.py")
    dot.node("evaluate", "evaluate.py")
    dot.node("demo", "demo.py")
    dot.node("ablations", "ablations.py")
    dot.node("monte_carlo", "monte_carlo.py")
    dot.node("escape_env", "escape_env.py")
    dot.node("policies", "policies.py")
    dot.node("utils", "utils.py")
    dot.node("importance_sampling", "importance_sampling.py")

    # Edges (dependencies)
    dot.edges([("train_mc", "monte_carlo"),
               ("train_mc", "escape_env"),
               ("train_mc", "policies"),
               ("train_mc", "utils")])

    dot.edges([("evaluate", "monte_carlo"),
               ("evaluate", "escape_env")])

    dot.edges([("demo", "monte_carlo"),
               ("demo", "escape_env")])

    dot.edges([("ablations", "train_mc"),
               ("ablations", "evaluate"),
               ("ablations", "monte_carlo"),
               ("ablations", "escape_env"),
               ("ablations", "importance_sampling")])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Render PNG
    dot.render(output_path, cleanup=True)
    print(f"✅ Dependency graph saved to {output_path}.png")


if __name__ == "__main__":
    generate_dependency_graph()
