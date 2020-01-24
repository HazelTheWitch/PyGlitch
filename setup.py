from setuptools import setup, find_packages

setup(
    name='pyglitch',
    description='"Glitchy" image effects for Python',
    author='Hazel Rella',
    author_email='hazelrella11@gmail.com',
    version='0.0.1',
    packages=find_packages(include=['pyglitch', 'pyglitch.*']),
    install_requires=[
        'numpy==1.18.1',
        'Pillow==7.0.0'
    ]
)
