import typer
from gseapyvis import plot_gsea

app = typer.Typer()
app.add_typer(plot_gsea.app)

if __name__ == "__main__":
    app()


