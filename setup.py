from setuptools import setup
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pybaycor',
    version=get_version("pybaycor/__init__.py"),    
    description='A package for Bayesian inference of correlations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pscicluna/pybaycor',
    project_urls={
        "Bug Tracker": "https://github.com/pscicluna/pybaycor/issues",
    },
    author='Peter Scicluna',
    author_email='peter.scicluna@eso.org',
    license='MIT',
    packages=['pybaycor'],
    install_requires=['numpy',
                      'matplotlib',
                      'seaborn',
                      'xarray',
                      'pymc3==3.10.0',
                      'arviz==0.11.0',
                      'xarray'
                      ],
    python_requires=">=3.7", 

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)
