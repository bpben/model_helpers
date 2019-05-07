from setuptools import setup

setup(name='model_helpers',
      packages=['model_helpers'],
      version='0.3',
      description='Tools for helping construct/train models',
      url='https://github.com/bpben/model_helpers',
      download_url='https://github.com/bpben/model_helpers/archive/0.1.tar.gz',
      author='BB',
      author_email='yksrotab@gmail.com',
      keywords =  'models ML tuning training',
      license='MIT',
      install_requires=['scikit-learn', 'pandas', 'xgboost'],
      python_requires='>=3.6',
      classifiers=['Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: MIT License',
      'Programming Language :: Python :: 3.6'])
