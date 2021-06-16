#include "gtest/gtest.h"
#include "korali.hpp"
#include "sample/sample.hpp"

namespace
{
 using namespace korali;

 TEST(Sample, cornerCases)
 {
  Sample s;
  std::function<void(Sample &)> fc = [](Sample &s){};
  _functionVector.push_back(&fc);
  ASSERT_NO_THROW(s.run(0));
  ASSERT_ANY_THROW(s.run(10));

  ASSERT_NO_THROW(s[0] = "Value");

  Sample s1;
  ASSERT_NO_THROW(s1["Key"] = "Value");
  ASSERT_NO_THROW(s1.get<std::string>("", 0, "Key"));
  ASSERT_ANY_THROW(s1.get<int>("", 0, "Key"));
  ASSERT_ANY_THROW(s1.get<int>("", 0, "Unknown"));
 }

} // namespace
