// Generated by gencpp from file zed_wrapper/start_remote_streamRequest.msg
// DO NOT EDIT!


#ifndef ZED_WRAPPER_MESSAGE_START_REMOTE_STREAMREQUEST_H
#define ZED_WRAPPER_MESSAGE_START_REMOTE_STREAMREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace zed_wrapper
{
template <class ContainerAllocator>
struct start_remote_streamRequest_
{
  typedef start_remote_streamRequest_<ContainerAllocator> Type;

  start_remote_streamRequest_()
    {
    }
  start_remote_streamRequest_(const ContainerAllocator& _alloc)
    {
  (void)_alloc;
    }





  enum {
    codec = 0u,
    port = 30000u,
    bitrate = 2000u,
    gop_size = -1,
  };

  static const uint8_t adaptative_bitrate;

  typedef boost::shared_ptr< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> const> ConstPtr;

}; // struct start_remote_streamRequest_

typedef ::zed_wrapper::start_remote_streamRequest_<std::allocator<void> > start_remote_streamRequest;

typedef boost::shared_ptr< ::zed_wrapper::start_remote_streamRequest > start_remote_streamRequestPtr;
typedef boost::shared_ptr< ::zed_wrapper::start_remote_streamRequest const> start_remote_streamRequestConstPtr;

// constants requiring out of line definition

   

   

   

   

   
   template<typename ContainerAllocator> const uint8_t
      start_remote_streamRequest_<ContainerAllocator>::adaptative_bitrate =
        
           0
        
        ;
   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace zed_wrapper

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'HasHeader': False, 'IsMessage': True}
// {}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "a6f55a6077162992b395e1b483a03367";
  }

  static const char* value(const ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xa6f55a6077162992ULL;
  static const uint64_t static_value2 = 0xb395e1b483a03367ULL;
};

template<class ContainerAllocator>
struct DataType< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "zed_wrapper/start_remote_streamRequest";
  }

  static const char* value(const ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
\n\
uint8 codec=0\n\
\n\
\n\
\n\
uint16 port=30000\n\
\n\
\n\
uint32 bitrate=2000\n\
\n\
\n\
\n\
\n\
\n\
int32 gop_size=-1\n\
\n\
\n\
\n\
\n\
\n\
bool adaptative_bitrate=False\n\
";
  }

  static const char* value(const ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream&, T)
    {}

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct start_remote_streamRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream&, const std::string&, const ::zed_wrapper::start_remote_streamRequest_<ContainerAllocator>&)
  {}
};

} // namespace message_operations
} // namespace ros

#endif // ZED_WRAPPER_MESSAGE_START_REMOTE_STREAMREQUEST_H
