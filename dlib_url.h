// http://dlib.net/dlib/server/server_http.cpp.html
// http://dlib.net/license.html
// Copyright (C) 2003  Davis E. King (davis@dlib.net)
// License: Boost Software License
// Boost Software License - Version 1.0 - August 17th, 2003
//
// Permission is hereby granted, free of charge, to any person or organization
// obtaining a copy of the software and accompanying documentation covered by
// this license (the "Software") to use, reproduce, display, distribute,
// execute, and transmit the Software, and to prepare derivative works of the
// Software, and to permit third-parties to whom the Software is furnished to
// do so, all subject to the following:
// The copyright notices in the Software and this entire statement, including
// the above license grant, this restriction and the following disclaimer,
// must be included in all copies of the Software, in whole or in part, and
// all derivative works of the Software, unless such copies or derivative
// works are solely in the form of machine-executable object code generated by
// a source language processor.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
// SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
// FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//

// Include headers from server_http.h:
#include <sstream>
#include <string>

namespace dlib {
inline unsigned char to_hex(unsigned char x)
{
  return x + (x > 9 ? ('A' - 10) : '0');
}

const inline std::string urlencode(const std::string& s)
{
  std::ostringstream os;

  for(std::string::const_iterator ci = s.begin(); ci != s.end(); ++ci)
  {
    if((*ci >= 'a' && *ci <= 'z') || (*ci >= 'A' && *ci <= 'Z') || (*ci >= '0' && *ci <= '9'))
    {  // allowed
      os << *ci;
    }
    else if(*ci == ' ')
    {
      os << '+';
    }
    else
    {
      os << '%' << to_hex(*ci >> 4) << to_hex(*ci % 16);
    }
  }

  return os.str();
}

inline unsigned char from_hex(unsigned char ch)
{
  if(ch <= '9' && ch >= '0')
    ch -= '0';
  else if(ch <= 'f' && ch >= 'a')
    ch -= 'a' - 10;
  else if(ch <= 'F' && ch >= 'A')
    ch -= 'A' - 10;
  else
    ch = 0;
  return ch;
}

const inline std::string urldecode(const std::string& str)
{
  using namespace std;
  string            result;
  string::size_type i;
  for(i = 0; i < str.size(); ++i)
  {
    if(str[i] == '+')
    {
      result += ' ';
    }
    else if(str[i] == '%' && str.size() > i + 2)
    {
      const unsigned char ch1 = from_hex(str[i + 1]);
      const unsigned char ch2 = from_hex(str[i + 2]);
      const unsigned char ch  = (ch1 << 4) | ch2;
      result += ch;
      i += 2;
    }
    else
    {
      result += str[i];
    }
  }
  return result;
}
}  // namespace dlib