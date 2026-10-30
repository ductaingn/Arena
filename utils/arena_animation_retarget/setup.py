from setuptools import find_packages, setup

package_name = 'arena_animation_retarget'

setup(
    name=package_name,
    packages=find_packages(where='.', include=[f'{package_name}*']),
    package_dir={'': '.'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    zip_safe=True,
    maintainer='ductaingn',
    maintainer_email='ductaingn.015203@gmail.com',
    description='Retarget SMPL-X body motion onto Arena 40-DOF human wire clips.',
    license='TODO',
    entry_points={
        'console_scripts': [
            'retarget = arena_animation_retarget.cli:main',
        ]
    },
)
