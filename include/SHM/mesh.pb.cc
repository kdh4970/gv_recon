// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: mesh.proto
// Protobuf C++ Version: 4.26.0-dev

#include "mesh.pb.h"

#include <algorithm>
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/extension_set.h"
#include "google/protobuf/wire_format_lite.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/generated_message_reflection.h"
#include "google/protobuf/reflection_ops.h"
#include "google/protobuf/wire_format.h"
#include "google/protobuf/generated_message_tctable_impl.h"
// @@protoc_insertion_point(includes)

// Must be included last.
#include "google/protobuf/port_def.inc"
PROTOBUF_PRAGMA_INIT_SEG
namespace _pb = ::google::protobuf;
namespace _pbi = ::google::protobuf::internal;
namespace _fl = ::google::protobuf::internal::field_layout;

inline constexpr Mesh::Impl_::Impl_(
    ::_pbi::ConstantInitialized) noexcept
      : vertices_{},
        triangles_{},
        _triangles_cached_byte_size_{0},
        class_info_(
            &::google::protobuf::internal::fixed_address_empty_string,
            ::_pbi::ConstantInitialized()),
        _cached_size_{0} {}

template <typename>
PROTOBUF_CONSTEXPR Mesh::Mesh(::_pbi::ConstantInitialized)
    : _impl_(::_pbi::ConstantInitialized()) {}
struct MeshDefaultTypeInternal {
  PROTOBUF_CONSTEXPR MeshDefaultTypeInternal() : _instance(::_pbi::ConstantInitialized{}) {}
  ~MeshDefaultTypeInternal() {}
  union {
    Mesh _instance;
  };
};

PROTOBUF_ATTRIBUTE_NO_DESTROY PROTOBUF_CONSTINIT
    PROTOBUF_ATTRIBUTE_INIT_PRIORITY1 MeshDefaultTypeInternal _Mesh_default_instance_;
static ::_pb::Metadata file_level_metadata_mesh_2eproto[1];
static constexpr const ::_pb::EnumDescriptor**
    file_level_enum_descriptors_mesh_2eproto = nullptr;
static constexpr const ::_pb::ServiceDescriptor**
    file_level_service_descriptors_mesh_2eproto = nullptr;
const ::uint32_t
    TableStruct_mesh_2eproto::offsets[] ABSL_ATTRIBUTE_SECTION_VARIABLE(
        protodesc_cold) = {
        ~0u,  // no _has_bits_
        PROTOBUF_FIELD_OFFSET(::Mesh, _internal_metadata_),
        ~0u,  // no _extensions_
        ~0u,  // no _oneof_case_
        ~0u,  // no _weak_field_map_
        ~0u,  // no _inlined_string_donated_
        ~0u,  // no _split_
        ~0u,  // no sizeof(Split)
        PROTOBUF_FIELD_OFFSET(::Mesh, _impl_.class_info_),
        PROTOBUF_FIELD_OFFSET(::Mesh, _impl_.vertices_),
        PROTOBUF_FIELD_OFFSET(::Mesh, _impl_.triangles_),
};

static const ::_pbi::MigrationSchema
    schemas[] ABSL_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
        {0, -1, -1, sizeof(::Mesh)},
};

static const ::_pb::Message* const file_default_instances[] = {
    &::_Mesh_default_instance_._instance,
};
const char descriptor_table_protodef_mesh_2eproto[] ABSL_ATTRIBUTE_SECTION_VARIABLE(
    protodesc_cold) = {
    "\n\nmesh.proto\"\?\n\004Mesh\022\022\n\nclass_info\030\001 \001(\t"
    "\022\020\n\010vertices\030\002 \003(\002\022\021\n\ttriangles\030\003 \003(\005b\006p"
    "roto3"
};
static ::absl::once_flag descriptor_table_mesh_2eproto_once;
const ::_pbi::DescriptorTable descriptor_table_mesh_2eproto = {
    false,
    false,
    85,
    descriptor_table_protodef_mesh_2eproto,
    "mesh.proto",
    &descriptor_table_mesh_2eproto_once,
    nullptr,
    0,
    1,
    schemas,
    file_default_instances,
    TableStruct_mesh_2eproto::offsets,
    file_level_metadata_mesh_2eproto,
    file_level_enum_descriptors_mesh_2eproto,
    file_level_service_descriptors_mesh_2eproto,
};

// This function exists to be marked as weak.
// It can significantly speed up compilation by breaking up LLVM's SCC
// in the .pb.cc translation units. Large translation units see a
// reduction of more than 35% of walltime for optimized builds. Without
// the weak attribute all the messages in the file, including all the
// vtables and everything they use become part of the same SCC through
// a cycle like:
// GetMetadata -> descriptor table -> default instances ->
//   vtables -> GetMetadata
// By adding a weak function here we break the connection from the
// individual vtables back into the descriptor table.
PROTOBUF_ATTRIBUTE_WEAK const ::_pbi::DescriptorTable* descriptor_table_mesh_2eproto_getter() {
  return &descriptor_table_mesh_2eproto;
}
// ===================================================================

class Mesh::_Internal {
 public:
};

Mesh::Mesh(::google::protobuf::Arena* arena)
    : ::google::protobuf::Message(arena) {
  SharedCtor(arena);
  // @@protoc_insertion_point(arena_constructor:Mesh)
}
inline PROTOBUF_NDEBUG_INLINE Mesh::Impl_::Impl_(
    ::google::protobuf::internal::InternalVisibility visibility, ::google::protobuf::Arena* arena,
    const Impl_& from)
      : vertices_{visibility, arena, from.vertices_},
        triangles_{visibility, arena, from.triangles_},
        _triangles_cached_byte_size_{0},
        class_info_(arena, from.class_info_),
        _cached_size_{0} {}

Mesh::Mesh(
    ::google::protobuf::Arena* arena,
    const Mesh& from)
    : ::google::protobuf::Message(arena) {
  Mesh* const _this = this;
  (void)_this;
  _internal_metadata_.MergeFrom<::google::protobuf::UnknownFieldSet>(
      from._internal_metadata_);
  new (&_impl_) Impl_(internal_visibility(), arena, from._impl_);

  // @@protoc_insertion_point(copy_constructor:Mesh)
}
inline PROTOBUF_NDEBUG_INLINE Mesh::Impl_::Impl_(
    ::google::protobuf::internal::InternalVisibility visibility,
    ::google::protobuf::Arena* arena)
      : vertices_{visibility, arena},
        triangles_{visibility, arena},
        _triangles_cached_byte_size_{0},
        class_info_(arena),
        _cached_size_{0} {}

inline void Mesh::SharedCtor(::_pb::Arena* arena) {
  new (&_impl_) Impl_(internal_visibility(), arena);
}
Mesh::~Mesh() {
  // @@protoc_insertion_point(destructor:Mesh)
  _internal_metadata_.Delete<::google::protobuf::UnknownFieldSet>();
  SharedDtor();
}
inline void Mesh::SharedDtor() {
  ABSL_DCHECK(GetArena() == nullptr);
  _impl_.class_info_.Destroy();
  _impl_.~Impl_();
}

const ::google::protobuf::MessageLite::ClassData*
Mesh::GetClassData() const {
  PROTOBUF_CONSTINIT static const ::google::protobuf::MessageLite::
      ClassDataFull _data_ = {
          {
              nullptr,  // OnDemandRegisterArenaDtor
              PROTOBUF_FIELD_OFFSET(Mesh, _impl_._cached_size_),
              false,
          },
          &Mesh::MergeImpl,
          &Mesh::kDescriptorMethods,
      };
  return &_data_;
}
PROTOBUF_NOINLINE void Mesh::Clear() {
// @@protoc_insertion_point(message_clear_start:Mesh)
  PROTOBUF_TSAN_WRITE(&_impl_._tsan_detect_race);
  ::uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  _impl_.vertices_.Clear();
  _impl_.triangles_.Clear();
  _impl_.class_info_.ClearToEmpty();
  _internal_metadata_.Clear<::google::protobuf::UnknownFieldSet>();
}

const char* Mesh::_InternalParse(
    const char* ptr, ::_pbi::ParseContext* ctx) {
  ptr = ::_pbi::TcParser::ParseLoop(this, ptr, ctx, &_table_.header);
  return ptr;
}


PROTOBUF_CONSTINIT PROTOBUF_ATTRIBUTE_INIT_PRIORITY1
const ::_pbi::TcParseTable<2, 3, 0, 23, 2> Mesh::_table_ = {
  {
    0,  // no _has_bits_
    0, // no _extensions_
    3, 24,  // max_field_number, fast_idx_mask
    offsetof(decltype(_table_), field_lookup_table),
    4294967288,  // skipmap
    offsetof(decltype(_table_), field_entries),
    3,  // num_field_entries
    0,  // num_aux_entries
    offsetof(decltype(_table_), field_names),  // no aux_entries
    &_Mesh_default_instance_._instance,
    ::_pbi::TcParser::GenericFallback,  // fallback
    #ifdef PROTOBUF_PREFETCH_PARSE_TABLE
    ::_pbi::TcParser::GetTable<::Mesh>(),  // to_prefetch
    #endif  // PROTOBUF_PREFETCH_PARSE_TABLE
  }, {{
    {::_pbi::TcParser::MiniParse, {}},
    // string class_info = 1;
    {::_pbi::TcParser::FastUS1,
     {10, 63, 0, PROTOBUF_FIELD_OFFSET(Mesh, _impl_.class_info_)}},
    // repeated float vertices = 2;
    {::_pbi::TcParser::FastF32P1,
     {18, 63, 0, PROTOBUF_FIELD_OFFSET(Mesh, _impl_.vertices_)}},
    // repeated int32 triangles = 3;
    {::_pbi::TcParser::FastV32P1,
     {26, 63, 0, PROTOBUF_FIELD_OFFSET(Mesh, _impl_.triangles_)}},
  }}, {{
    65535, 65535
  }}, {{
    // string class_info = 1;
    {PROTOBUF_FIELD_OFFSET(Mesh, _impl_.class_info_), 0, 0,
    (0 | ::_fl::kFcSingular | ::_fl::kUtf8String | ::_fl::kRepAString)},
    // repeated float vertices = 2;
    {PROTOBUF_FIELD_OFFSET(Mesh, _impl_.vertices_), 0, 0,
    (0 | ::_fl::kFcRepeated | ::_fl::kPackedFloat)},
    // repeated int32 triangles = 3;
    {PROTOBUF_FIELD_OFFSET(Mesh, _impl_.triangles_), 0, 0,
    (0 | ::_fl::kFcRepeated | ::_fl::kPackedInt32)},
  }},
  // no aux_entries
  {{
    "\4\12\0\0\0\0\0\0"
    "Mesh"
    "class_info"
  }},
};

::uint8_t* Mesh::_InternalSerialize(
    ::uint8_t* target,
    ::google::protobuf::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:Mesh)
  ::uint32_t cached_has_bits = 0;
  (void)cached_has_bits;

  // string class_info = 1;
  if (!this->_internal_class_info().empty()) {
    const std::string& _s = this->_internal_class_info();
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
        _s.data(), static_cast<int>(_s.length()), ::google::protobuf::internal::WireFormatLite::SERIALIZE, "Mesh.class_info");
    target = stream->WriteStringMaybeAliased(1, _s, target);
  }

  // repeated float vertices = 2;
  if (this->_internal_vertices_size() > 0) {
    target = stream->WriteFixedPacked(2, _internal_vertices(), target);
  }

  // repeated int32 triangles = 3;
  {
    int byte_size = _impl_._triangles_cached_byte_size_.Get();
    if (byte_size > 0) {
      target = stream->WriteInt32Packed(
          3, _internal_triangles(), byte_size, target);
    }
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target =
        ::_pbi::WireFormat::InternalSerializeUnknownFieldsToArray(
            _internal_metadata_.unknown_fields<::google::protobuf::UnknownFieldSet>(::google::protobuf::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:Mesh)
  return target;
}

::size_t Mesh::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:Mesh)
  ::size_t total_size = 0;

  ::uint32_t cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated float vertices = 2;
  {
    std::size_t data_size = std::size_t{4} *
        ::_pbi::FromIntSize(this->_internal_vertices_size())
    ;
    std::size_t tag_size = data_size == 0
        ? 0
        : 1 + ::_pbi::WireFormatLite::Int32Size(
                            static_cast<int32_t>(data_size))
    ;
    total_size += tag_size + data_size;
  }
  // repeated int32 triangles = 3;
  {
    std::size_t data_size = ::_pbi::WireFormatLite::Int32Size(
        this->_internal_triangles())
    ;
    _impl_._triangles_cached_byte_size_.Set(::_pbi::ToCachedSize(data_size));
    std::size_t tag_size = data_size == 0
        ? 0
        : 1 + ::_pbi::WireFormatLite::Int32Size(
                            static_cast<int32_t>(data_size))
    ;
    total_size += tag_size + data_size;
  }
  // string class_info = 1;
  if (!this->_internal_class_info().empty()) {
    total_size += 1 + ::google::protobuf::internal::WireFormatLite::StringSize(
                                    this->_internal_class_info());
  }

  return MaybeComputeUnknownFieldsSize(total_size, &_impl_._cached_size_);
}


void Mesh::MergeImpl(::google::protobuf::MessageLite& to_msg, const ::google::protobuf::MessageLite& from_msg) {
  auto* const _this = static_cast<Mesh*>(&to_msg);
  auto& from = static_cast<const Mesh&>(from_msg);
  // @@protoc_insertion_point(class_specific_merge_from_start:Mesh)
  ABSL_DCHECK_NE(&from, _this);
  ::uint32_t cached_has_bits = 0;
  (void) cached_has_bits;

  _this->_internal_mutable_vertices()->MergeFrom(from._internal_vertices());
  _this->_internal_mutable_triangles()->MergeFrom(from._internal_triangles());
  if (!from._internal_class_info().empty()) {
    _this->_internal_set_class_info(from._internal_class_info());
  }
  _this->_internal_metadata_.MergeFrom<::google::protobuf::UnknownFieldSet>(from._internal_metadata_);
}

void Mesh::CopyFrom(const Mesh& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:Mesh)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

PROTOBUF_NOINLINE bool Mesh::IsInitialized() const {
  return true;
}

void Mesh::InternalSwap(Mesh* PROTOBUF_RESTRICT other) {
  using std::swap;
  auto* arena = GetArena();
  ABSL_DCHECK_EQ(arena, other->GetArena());
  _internal_metadata_.InternalSwap(&other->_internal_metadata_);
  _impl_.vertices_.InternalSwap(&other->_impl_.vertices_);
  _impl_.triangles_.InternalSwap(&other->_impl_.triangles_);
  ::_pbi::ArenaStringPtr::InternalSwap(&_impl_.class_info_, &other->_impl_.class_info_, arena);
}

::google::protobuf::Metadata Mesh::GetMetadata() const {
  return ::_pbi::AssignDescriptors(&descriptor_table_mesh_2eproto_getter,
                                   &descriptor_table_mesh_2eproto_once,
                                   file_level_metadata_mesh_2eproto[0]);
}
// @@protoc_insertion_point(namespace_scope)
namespace google {
namespace protobuf {
}  // namespace protobuf
}  // namespace google
// @@protoc_insertion_point(global_scope)
PROTOBUF_ATTRIBUTE_INIT_PRIORITY2
static ::std::false_type _static_init_ PROTOBUF_UNUSED =
    (::_pbi::AddDescriptors(&descriptor_table_mesh_2eproto),
     ::std::false_type{});
#include "google/protobuf/port_undef.inc"