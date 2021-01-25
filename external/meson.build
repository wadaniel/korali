# request the compiler object
_c = meson.get_compiler('c')
_cpp = meson.get_compiler('cpp')

# get the build environment
build_env = environment()
build_env.set('CC', _c.cmd_array()[0])
build_env.set('CXX', _cpp.cmd_array()[0])

# required external libraries.  Dictionary values are lists of compiler object
# and fallback installer script that will install the dependency in
# MESON_INSTALL_PREFIX
external_required = {
  'gsl': [_cpp, 'install_gsl.sh'],
  # 'co': [_c, 'install_libco.sh'],
  }

foreach libname, fallback : external_required
  _lib = fallback[0].find_library(libname, required: false)
  # _lib = dependency(libname, required: false)
  run_command(fallback[1], check: true, env: build_env)
  if not _lib.found()
    run_command(fallback[1], check: true, env: build_env)
    _d = declare_dependency(include_directories: 'prefix'/'include')
    korali_deps += _d
  else
    korali_deps += _lib
  endif
endforeach
