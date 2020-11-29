from setuptools import setup

setup(name='gomc-wrapper',
      version='0.01',
      description='Integrate along any coexistence curve using the Gibbs-Duhem equation',
      url='http://github.com/evenmn/gibbs-duhem-integrator',
      author='Even Marius Nordhagen',
      author_email='evenmn@fys.uio.no',
      license='MIT',
      packages=['gibbsduhem'],
      include_package_data=True,
      zip_safe=False)
