from fabric_model.gui import plotly_main
from fabric_model.gui.dash_backend import app as application
app = application.server

def main():
    #app.run_server()
    plotly_main.main()
    ...

if __name__ == '__main__':
    main()