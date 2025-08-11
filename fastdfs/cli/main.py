import typer

from . import preprocess

app = typer.Typer(pretty_exceptions_enable=False, help="FastDFS - Deep Feature Synthesis for Tabular Data")

# Add preprocess command
app.command(name="preprocess")(preprocess.preprocess)

if __name__ == '__main__':
    app()
