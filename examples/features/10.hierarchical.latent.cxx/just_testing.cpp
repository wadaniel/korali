#include <map>
#include <vector.h>

int main()
{
  std::map<int, std::vector<int>> my_map;

  for (int i = 0; i < 5; i++)
  {
    my_map[i * i] = std::vector<int>();
    my_map[i * i].clear();
    my_map[i * i].push_back(i);
  }
  for (int i = 0; i < 5; i++)
  {
    assert(std::find(my_map[i * i].begin(), my_map[i * i].end(), i) != my_map[i * i].end());
  }
}
