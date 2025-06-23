import os
import argparse
import textwrap


def scaffold_plugin(stage: str, plugin_name: str, base_module: str, namespace: str, target_dir: str):
    plugin_dir = os.path.join(target_dir, f"{namespace}.{stage}")
    os.makedirs(plugin_dir, exist_ok=True)

    init_path = os.path.join(plugin_dir, "__init__.py")
    plugin_path = os.path.join(plugin_dir, f"{plugin_name}.py")

    class_name = f"{plugin_name.title().replace('_', '')}{stage.title()}"

    # Write __init__.py
    with open(init_path, "w") as f:
        f.write("")

    # Write plugin class file
    with open(plugin_path, "w") as f:
        f.write(textwrap.dedent(f"""
            import logging
            from {base_module} import {stage.title()}Base

            class {class_name}({stage.title()}Base):
                \"\"\"{plugin_name} plugin for {stage} stage.\"\"\"

                def __init__(self, ctx=None):
                    \"\"\"Initialize the plugin with optional context.
                    
                    Args:
                        ctx: Optional context dictionary containing plugin configuration.
                             Can include plugin-specific parameters.
                    \"\"\"
                    super().__init__(ctx)
                    self.logger = logging.getLogger(__name__)
                    
                    # Example of accessing configuration from context
                    # Replace with actual parameters needed for your plugin
                    self.param = 0.5  # Default value
                    if self.ctx and 'param' in self.ctx:
                        self.param = self.ctx['param']

                def {stage}(self, *args, **kwargs):
                    \"\"\"Your implementation goes here.
                    
                    This method should implement the functionality for the {stage} stage.
                    The parameters will depend on the specific stage.
                    
                    Returns:
                        The result of the {stage} operation.
                    \"\"\"
                    self.logger.info("Running {plugin_name} plugin.")
                    # Example implementation - replace with actual code
                    return args[0]  # placeholder
        """))

    # Print setup.cfg block
    entry = textwrap.dedent(f"""
    [options.entry_points]
    {namespace}.{stage} =
        {plugin_name} = {namespace}.{stage}.{plugin_name}:{class_name}
    """)

    print("✅ Plugin scaffold created:")
    print(f"- Directory: {plugin_dir}")
    print(f"- Class: {class_name}")
    print(f"- Entry point block (add to setup.cfg):\n{entry}")


def main():
    parser = argparse.ArgumentParser(
        description="Scaffold a new plugin for the GA system using Stevedore."
    )

    parser.add_argument("stage", help="Plugin stage (e.g. mutate, select, fitness)")
    parser.add_argument("plugin_name", help="Name of the plugin (e.g. bitflip, roulette)")

    parser.add_argument(
        "--base-module",
        default="pangadfs.base",
        help="Python import path to base interfaces (default: pangadfs.base)"
    )
    parser.add_argument(
        "--namespace",
        default="pangadfs",
        help="Stevedore namespace prefix (default: pangadfs)"
    )
    parser.add_argument(
        "--target-dir",
        default=".",
        help="Where to create the plugin module (default: current directory)"
    )

    args = parser.parse_args()

    scaffold_plugin(
        stage=args.stage,
        plugin_name=args.plugin_name,
        base_module=args.base_module,
        namespace=args.namespace,
        target_dir=args.target_dir,
    )


if __name__ == "__main__":
    main()
