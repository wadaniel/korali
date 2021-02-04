project('gsl', 'c',
  version: '2.6',
  meson_version : '>=0.56.0',
  default_options: ['b_ndebug=if-release']
)

# fetch the source
_gsl_version = meson.project_version()
_r = run_command('._fetch.sh', [_gsl_version])
if _r.returncode() != 0
  error('Failed to obtain gsl source for version ' + _gsl_version)
endif

# requires meson >= 0.56.0
mod = import('unstable_external_project')

lib_type = get_option('default_library')
lib_opt = [
  '--prefix=@PREFIX@',
  '--libdir=@PREFIX@/@LIBDIR@',
  '--includedir=@PREFIX@/@INCLUDEDIR@',
  '--with-pic',
]
if lib_type == 'shared'
  lib_opt += '--enable-shared=yes'
  lib_opt += '--enable-static=no'
elif lib_type == 'static'
  lib_opt += '--enable-shared=no'
  lib_opt += '--enable-static=yes'
endif

p = mod.add_project('configure',
  configure_options: lib_opt,
)

gsl_dep = p.dependency('gsl')
