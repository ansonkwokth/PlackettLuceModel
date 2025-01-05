from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os

class CustomInstallCommand(install):
    def run(self):
        install.run(self)  # Run the default install command
        # Now, run the example data generation script
        print("Running post-installation script to generate example data...")
        data_gen_script = os.path.join(os.path.dirname(__file__), 'scripts', 'generate_example_data')
        subprocess.check_call(['python', data_gen_script])


setup(
    name="plackett_luce",
    version="0.1.0",
    description="Ranking model",
    author="Anson",
    author_email="ansonwos@gmail.com",
    packages=find_packages(where="plackett_luce"),
    package_dir={"": "plackett_luce"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.26",
        "torch>=2.5",
    ],
    cmdclass={
        'install': CustomInstallCommand,  # Override the install command to run our custom one
    },
    include_package_data=True,
    package_data={
        '': ['data/example_data/*'],  # Include the example data directory in the package
    },
)
