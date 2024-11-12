# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PreScyent'
copyright = '2024, Alexis Biver'
author = 'Alexis Biver'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["myst_parser", 'sphinx_markdown', 'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


import importlib

def add_default_values(app, what, name, obj, options, lines):
    # only process attributes for pydantic configs autodocs
    if what == "attribute":
        # the name is the fully-qualified name
        parts = name.split('.')
        attribute_name = parts[-1]
        class_name = parts[-2]
        module_name = '.'.join(parts[:-2])
        try:
            # dynamically import the class
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            default_value = getattr(cls, attribute_name, None)
            # If dfefault wasn't found, get it from BaseModel.model_fields
            if default_value is None:
                pydantic_fiels_infos = cls.model_fields.get(attribute_name)
                required = pydantic_fiels_infos.is_required()
                lines.append(f"**Required:** {required}")
                if not required:
                    default_value = pydantic_fiels_infos.get_default()
                    lines.append("")
                    lines.append(f"**Default:** {default_value}")
        except (ImportError, AttributeError) as e:
            print(f"Error loading {module_name}.{class_name}: {e}")

def skip_member(app, what, name, obj, skip, options):
    # Skip inherited members if they are from a certain base class
    if what == "class" and "BaseConfig." in str(obj):
        return True  # Skip all members from BaseConfig
    if what == "class" and "BaseModel." in str(obj):
        return True  # Skip all members from BaseConfig
    if obj.__class__.__name__ == "property":
        return True  # Skip all properties
    return skip

def setup(app):
    print("Connecting event handlers...")
    app.connect("autodoc-skip-member", skip_member)
    app.connect('autodoc-process-docstring', add_default_values)
    print("Event handlers connected.")
