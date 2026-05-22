from setuptools import setup, find_packages

setup(
    name="ayudante",
    version="1.0.0",
    description="Kit de herramientas reutilizables para EDA y Machine Learning",
    author="Silvia",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scikit-learn>=1.3",
        "scipy>=1.10",
    ],
)
