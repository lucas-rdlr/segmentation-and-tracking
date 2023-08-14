import os

def init():
    """
    Creates a basic tree folder for a Machine Learning project with the name and path
    asked when running the module.
    Since input() function converts everything to str data type, the name and path
    should be given without quotes.
    If no path is given, the folder will be created in the current path.
    """

    cwd = os.getcwd()
    print('Current working directory:', cwd)

    path = input("Path to project [None for cd]: ") or None
    name = input("Project's name: ")
    
    if path is not None:
        os.chdir(path)

    os.mkdir(name)
    os.chdir(name)

    data = ['external', 'internal', 'processed', 'raw']
    os.mkdir('data')
    for folder in data:
        os.mkdir(f'data/{folder}')
    
    src = ['data', 'features', 'models', 'visualization']
    os.mkdir('src')
    for folder in src:
        os.mkdir(f'src/{folder}')

    os.mkdir('reports')
    os.mkdir('reports/figures')

    os.mkdir('references')
    os.mkdir('models')
    os.mkdir('notebooks')
    os.mkdir('experiments')
    
    fp = open('README.md', 'x')
    fp.close()

    fp = open('requirements.txt', 'x')
    fp.close()

    fp = open('notes.txt', 'x')
    fp.close()

    fp = open('src/__init__.py', 'x')
    fp.close()

    return

init()