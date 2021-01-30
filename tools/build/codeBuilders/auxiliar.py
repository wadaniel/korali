from pathlib import Path
import json
from . import header_builders as hb

def pathSplitAtDir( path, dir ):
  ''' Return the path in the `path` variable up to the dir folder.
      e.g. if path=Path('/a/b/c/d') and dir='c', returns Path('a/b/c')

      Parameters
      ----------
      path : pathlib.Path object

      dir : string
  '''
  p = Path('/')
  for i in path.parts:
    p = p.joinpath(i)
    if i==dir: break
  return p


def loadModuleConfiguration( configFilePath, templateFilePath ):

  moduleConfig = json.loads( configFilePath.read_text() )
  modulesPath = pathSplitAtDir(configFilePath,'modules')

  moduleConfig['Name'] = str( configFilePath.parent.parts[-1] )
  moduleConfig['Class Name'] = moduleConfig['Module Data']['Class Name']
  moduleConfig['Parent Class Name'] = moduleConfig['Module Data']['Parent Class Name']
  moduleConfig['Namespace'] = moduleConfig['Module Data']['Namespace']
  moduleConfig['Module Type'] = str( Path(*configFilePath.relative_to(modulesPath).parts[1:-1]) )
  moduleConfig['Relative Path'] = str( configFilePath.parent.relative_to(modulesPath) )

  guard = '_'.join(moduleConfig['Namespace'])
  guard = '_' + guard.upper() + '_' + moduleConfig['Class Name'].upper() + '_'
  moduleConfig['Include Guard'] = guard

  return moduleConfig



def replaceKeys( moduleConfig, codeString ):
  codeString = codeString.replace( '@startIncludeGuard', hb.startIncludeGuard(moduleConfig) )
  codeString = codeString.replace( '@endIncludeGuard', hb.endIncludeGuard(moduleConfig) )

  codeString = codeString.replace( '@className', moduleConfig['Class Name'] )
  codeString = codeString.replace( '@parentClassName', moduleConfig['Parent Class Name'] )

  codeString = codeString.replace( '@startNamespace', hb.startNamespace(moduleConfig) )
  codeString = codeString.replace( '@endNamespace', hb.endNamespace(moduleConfig) )

  return codeString


