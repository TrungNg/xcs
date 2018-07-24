from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# SRC_DIR = "cysrc"
# PACKAGES = [SRC_DIR]
#
# ext_algorithm = Extension(SRC_DIR + ".xcs_algorithm",
#                   [SRC_DIR + "/xcs_algorithm.pyx"]
#                   )
#
# ext_class_accuracy = Extension(SRC_DIR + ".xcs_class_accuracy",
#                   [SRC_DIR + "/xcs_class_accuracy.pyx"]
#                   )
#
# ext_classifier = Extension(SRC_DIR + ".xcs_classifier",
#                   [SRC_DIR + "/xcs_classifier.pyx"]
#                   )
#
# ext_classifierset = Extension(SRC_DIR + ".xcs_classifierset",
#                   [SRC_DIR + "/xcs_classifierset.pyx"]
#                   )
#
# ext_config_parser = Extension(SRC_DIR + ".xcs_config_parser",
#                   [SRC_DIR + "/xcs_config_parser.pyx"]
#                   )
#
# ext_constants = Extension(SRC_DIR + ".xcs_constants",
#                   [SRC_DIR + "/xcs_constants.pyx"]
#                   )
#
# ext_data_management = Extension(SRC_DIR + ".xcs_data_management",
#                   [SRC_DIR + "/xcs_data_management.pyx"]
#                   )
# ext_offline_environment = Extension(SRC_DIR + ".xcs_offline_environment",
#                   [SRC_DIR + "/xcs_offline_environment.pyx"]
#                   )
#
# ext_online_environment = Extension(SRC_DIR + ".xcs_online_environment",
#                   [SRC_DIR + "/xcs_online_environment.pyx"]
#                   )
#
# ext_outputfile_manager = Extension(SRC_DIR + ".xcs_outputfile_manager",
#                   [SRC_DIR + "/xcs_outputfile_manager.pyx"]
#                   )
#
# ext_prediction = Extension(SRC_DIR + ".xcs_prediction",
#                   [SRC_DIR + "/xcs_prediction.pyx"]
#                   )
#
# ext_timer = Extension(SRC_DIR + ".xcs_timer",
#                   [SRC_DIR + "/xcs_timer.pyx"]
#                   )
#
# ext_runner = Extension(SRC_DIR + ".xcs_run",
#                   [SRC_DIR + "/xcs_run.pyx"]
#                   )

#EXTENSIONS = cythonize([ext_constants, ext_config_parser, ext_data_management, ext_offline_environment, ext_online_environment, ext_outputfile_manager, ext_timer, ext_prediction, ext_classifier, ext_classifierset, ext_class_accuracy, ext_algorithm])

#ext = Extension("xcs",["cysrc/*.pyx","randgen.c"])

# setup(
    #ext_modules = cythonize(["cysrc/*.pyx"]),
    # ext_modules = cythonize(["cysrc/randgen.c","cysrc/*.pyx"])
                              #language="c++",)
# )

ext_modules = [
    Extension("*",
              sources=["cysrc/*.pyx","cysrc/rand.c"]
              )
]

setup(name="xcs",
      ext_modules=cythonize(ext_modules))