import setuptools


setuptools.setup(
    name='aou_tool',
    version='0.0.1',
    author='Clarence Jiang',
    author_email='yj2737@columbia.edu',
    description='aou tool ',

    url='https://github.com/ClarenceJiang71/aou_tool',
    packages=['aou_tool'],
    install_requires=['requests', 'subprocess', 'numpy', 'pandas', 'hail', 'seaborn', 'matplotlib'],
)