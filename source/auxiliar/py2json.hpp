/***************************************************************************
* Copyright (c) 2019, Martin Renou
* Modified by Sergio Martin to support Function storing                    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
****************************************************************************/

#ifndef PYBIND_JSON_HPP
#define PYBIND_JSON_HPP

#include <string>
#include <vector>
#include "auxiliar/json.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"
#include "engine/sample/sample.hpp"

namespace nlohmann
{
    template <>
    struct adl_serializer<pybind11::object>
    {
        static pybind11::object from_json(const json& j);
        static void to_json(json& j, const pybind11::object& obj);
    };

    namespace detail
    {
        inline pybind11::object from_json_impl(const json& j)
        {
            if (j.is_null())
            {
                return pybind11::none();
            }
            else if (j.is_boolean())
            {
                return pybind11::bool_(j.get<bool>());
            }
            else if (j.is_number())
            {
                double number = j.get<double>();
                if (number == std::floor(number))
                {
                    return pybind11::int_(j.get<long>());
                }
                else
                {
                    return pybind11::float_(number);
                }
            }
            else if (j.is_string())
            {
                return pybind11::str(j.get<std::string>());
            }
            else if (j.is_array())
            {
                pybind11::list obj;
                for (const auto& el : j)
                {
                    obj.attr("append")(from_json_impl(el));
                }
                return obj;
            }
            else // Object
            {
                pybind11::dict obj;
                for (json::const_iterator it = j.cbegin(); it != j.cend(); ++it)
                {
                    obj[pybind11::str(it.key())] = from_json_impl(it.value());
                }
                return obj;
            }
        }

        inline json to_json_impl(const pybind11::handle& obj)
        {
            if (obj.is_none())
            {
                return nullptr;
            }
            if (pybind11::isinstance<pybind11::function>(obj))
            {
               return (std::uint64_t) new std::function<void(korali::Sample&)>(obj.cast<std::function<void(korali::Sample&)>>());
            }
            if (pybind11::isinstance<pybind11::bool_>(obj))
            {
                return obj.cast<bool>();
            }
            if (pybind11::isinstance<pybind11::int_>(obj))
            {
                return obj.cast<long>();
            }
            if (pybind11::isinstance<pybind11::float_>(obj))
            {
                return obj.cast<double>();
            }
            if (pybind11::isinstance<pybind11::str>(obj))
            {
                return obj.cast<std::string>();
            }
            if (pybind11::isinstance<pybind11::tuple>(obj) || pybind11::isinstance<pybind11::list>(obj))
            {
                auto out = json::array();
                for (const pybind11::handle& value : obj)
                {
                    out.push_back(to_json_impl(value));
                }
                return out;
            }
            if (pybind11::isinstance<pybind11::dict>(obj))
            {
                auto out = json::object();
                for (const pybind11::handle& key : obj)
                {
                    out[pybind11::str(key).cast<std::string>()] = to_json_impl(obj[key]);
                }
                return out;
            }
            throw std::runtime_error("to_json not implemented for this type of object: " + obj.cast<std::string>());
        }
    }

    inline pybind11::object adl_serializer<pybind11::object>::from_json(const json& j)
    {
        return detail::from_json_impl(j);
    }

    inline void adl_serializer<pybind11::object>::to_json(json& j, const pybind11::object& obj)
    {
        j = detail::to_json_impl(obj);
    }

}

#endif
