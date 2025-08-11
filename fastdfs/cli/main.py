import typer
from ..utils.logging_config import configure_logging

from . import preprocess

app = typer.Typer(pretty_exceptions_enable=False, help="FastDFS - Deep Feature Synthesis for Tabular Data")

# Configure logging at startup
configure_logging()

# Add preprocess command
app.command(name="preprocess")(preprocess.preprocess)

if __name__ == '__main__':
    app()
