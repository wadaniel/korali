#ifndef _KORALI_MODULE_HPP_
#define _KORALI_MODULE_HPP_

namespace Korali {

class Module
{
 public:

  static Module* getModule(nlohmann::json& js);

  virtual bool checkTermination() { return false; }
  virtual void initialize() { }
  virtual void finalize() { }
  virtual std::string getType() = 0;

  // Module serialization
  virtual void getConfiguration(nlohmann::json& js) = 0;
  virtual void setConfiguration(nlohmann::json& js) = 0;
};

} // namespace Korali

#endif // _KORALI_MODULE_HPP_
