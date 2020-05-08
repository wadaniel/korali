/** \file
* @brief Contains auxiliar code for file system (files and folders) manipulation.
******************************************************************************/

#ifndef _AUXILIAR_FS_HPP_
#define _AUXILIAR_FS_HPP_

#include <sys/stat.h>
#include <dirent.h>
#include <vector>

namespace korali
{

 /**
 * @brief Creates a new folder and builds the entire path, if necessary.
 * @param dirPath relative path to the new folder.
 */
 static void mkdir(const std::string dirPath)
 {
         char tmp[256];
         char *p = NULL;
         size_t len;

         snprintf(tmp, sizeof(tmp),"%s",dirPath.c_str());
         len = strlen(tmp);
         if(tmp[len - 1] == '/')
                 tmp[len - 1] = 0;
         for(p = tmp + 1; *p; p++)
                 if(*p == '/') {
                         *p = 0;
                         ::mkdir(tmp, S_IRWXU);
                         *p = '/';
                 }
         ::mkdir(tmp, S_IRWXU);
 }

 /**
 * @brief Lists all files within within a given folder path
 * @param dirPath relative path to the folder to list.
 * @return A list with the path of all files found.
 */
 static std::vector<std::string> listDirFiles(const std::string dirPath)
 {
     struct dirent *entry = nullptr;
     std::vector<std::string> dirList;

     DIR *dp = nullptr;

     dp = opendir(dirPath.c_str());
     if (dp != nullptr)  while ((entry = readdir(dp)))
     {
      std::string filePath = dirPath;
      filePath += "/";
      filePath += entry->d_name;
      dirList.push_back(filePath);
     }
     closedir(dp);

     return dirList;
 }

 /**
  * @brief Checks if directory exists
  * @param dirPath relative path to the folder to list.
  * @return true, if the directory exists, or; false, if it does not.
  */
 static bool dirExists(const std::string dirPath)
 {
   DIR* dir = opendir(dirPath.c_str());
   if (dir)
   {
       closedir(dir);
       return true;
   }

   return false;
 }
}

#endif // _AUXILIAR_FS_HPP_
