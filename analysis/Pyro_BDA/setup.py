from setuptools import setup, find_namespace_packages

setup(name='probabll.Pyro_BDA',
      version='1.0',
      description='BDA models',
      author='Probabll',
      author_email='w.aziz@uva.nl',
      url='https://github.com/probabll/bda',
      packages=find_namespace_packages(include=['probabll.*']),
      python_requires='>=3.6',
      include_package_data=True
)
