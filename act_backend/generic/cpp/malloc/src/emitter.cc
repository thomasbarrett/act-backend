#include "emitter.h"

namespace emitter {

MetaDataInfo::MetaDataInfo(nlohmann::json &j) {
  addr = 0;
  if (j.contains("addr") && j["addr"].is_number_integer())
    addr = j["addr"].get<int64>();
  else
    throw std::runtime_error("addr is required in metadata.json");

  shape.clear();
  if (j.contains("shape") && j["shape"].is_array()) {
    for (const auto &s : j["shape"]) {
      if (s.is_number_integer())
        shape.push_back(s.get<int64>());
      else
        throw std::runtime_error("shape must be an array of integers");
    }
  } else
    throw std::runtime_error("shape is required in metadata.json");

  if (j.contains("dtype") && j["dtype"].is_string())
    dtype = j["dtype"].get<std::string>();
  else
    throw std::runtime_error("dtype is required in metadata.json");
}

namespace {

// Element width in bytes for a Dtype as Rust's Debug prints it.
int64 dtype_width(const std::string &d) {
  if (d == "U8" || d == "I8") return 1;
  if (d == "U16" || d == "I16" || d == "FP16" || d == "BF16") return 2;
  if (d == "U32" || d == "I32" || d == "FP32") return 4;
  if (d == "U64" || d == "I64" || d == "FP64") return 8;
  std::cerr << "Error: unknown constant dtype " << d << std::endl;
  assert(false && "unknown constant dtype");
  return 0;
}

std::string jnp_dtype(const std::string &d) {
  if (d == "U8") return "jnp.uint8";
  if (d == "I8") return "jnp.int8";
  if (d == "U16") return "jnp.uint16";
  if (d == "I16") return "jnp.int16";
  if (d == "U32") return "jnp.uint32";
  if (d == "I32") return "jnp.int32";
  if (d == "U64") return "jnp.uint64";
  if (d == "I64") return "jnp.int64";
  if (d == "FP16") return "jnp.float16";
  if (d == "FP32") return "jnp.float32";
  if (d == "FP64") return "jnp.float64";
  if (d == "BF16") return "jnp.bfloat16";
  assert(false && "unknown constant dtype");
  return "";
}

// Pull value and dtype out of a representation whose only leaf is a single
// `constant[value='V',dtype='D']`. Returns false if it holds anything else,
// including a second constant -- then it is not a splat.
bool parse_splat_constant(const std::string &repr, std::string &value,
                          std::string &dtype) {
  const std::string tag = "constant[value='";
  size_t start = repr.find(tag);
  if (start == std::string::npos) return false;
  if (repr.find(tag, start + 1) != std::string::npos) return false;

  size_t vpos = start + tag.size();
  size_t vend = repr.find('\'', vpos);
  if (vend == std::string::npos) return false;
  value = repr.substr(vpos, vend - vpos);

  const std::string dtag = "dtype='";
  size_t dpos = repr.find(dtag, vend);
  if (dpos == std::string::npos) return false;
  dpos += dtag.size();
  size_t dend = repr.find('\'', dpos);
  if (dend == std::string::npos) return false;
  dtype = repr.substr(dpos, dend - dpos);
  return !value.empty() && !dtype.empty();
}

} // namespace

std::string MetaDataInfo::str() const {
  std::ostringstream oss;
  oss << "{'addr': " << addr << ", 'shape': (";
  for (size_t i = 0; i < shape.size(); i++) {
    oss << shape[i];
    if (i != shape.size() - 1)
      oss << ", ";
  }
  oss << "), 'dtype': " << dtype;
  if (value != "")
    oss << ", 'value': " << value;
  oss << "}";
  return oss.str();
}

MetaData::MetaData(
    std::ifstream &metadata_path,
    const std::unordered_map<std::string, Tensor *> &tensor_map,
    const std::unordered_map<Tensor *, std::string> &known_constants)
    : max_hbm_size(HBM_SIZE) {
  try {
    if (!metadata_path)
      throw std::runtime_error("failed to open metadata.json");

    nlohmann::json j;
    metadata_path.clear();
    metadata_path.seekg(0);
    metadata_path >> j;

    if (j.contains("module_name") && j["module_name"].is_string()) {
      this->module_name = j["module_name"].get<std::string>();
    } else
      throw std::runtime_error("module_name is required in metadata.json");

    if (j.contains("input") && j["input"].is_array()) {
      for (const auto &entry : j["input"]) {
        input_info.emplace_back(
            MetaDataInfo(const_cast<nlohmann::json &>(entry)));
      }
    } else
      throw std::runtime_error("input is required in metadata.json");

    if (j.contains("output") && j["output"].is_array()) {
      if (j["output"].size() != 1)
        throw std::runtime_error("supports only one output in metadata.json");

      for (const auto &entry : j["output"]) {
        output_info.emplace_back(
            MetaDataInfo(const_cast<nlohmann::json &>(entry)));
      }
    } else
      throw std::runtime_error("output is required in metadata.json");
  } catch (const std::exception &e) {
    std::cerr << "Warning: failed to parse metadata.json: " << e.what()
              << std::endl;
    assert(false && "nlohmann json parse error");
  }

  for (const auto &pair : tensor_map) {
    auto *tensor = pair.second;
    if (tensor->type == Tensor::CONSTANT) {
      if (known_constants.find(tensor) == known_constants.end()) {
        std::cerr << "Warning: constant tensor " << tensor->get_name()
                  << " not found in known constants." << std::endl;
        assert(false && "unknown constant tensor");
      }
      const std::string &repr = known_constants.at(tensor);

      if (repr == "reshape[shape='256'](bitcvt(eye[ttype='16,16,I8']()))") {
        constant_info.emplace_back(MetaDataInfo(
            tensor->get_offsets()[0]->Min(), tensor->get_sizes(), "jnp.uint8",
            "jnp.reshape(jnp.eye(16, dtype=jnp.int8).astype(jnp.uint8), "
            "(256,))"));
        continue;
      }

      // A splat: some arrangement of broadcast/reshape/convert around a single
      // constant. That covers every literal a real kernel carries -- 1/D, an
      // epsilon, an attention scale. Emit it as the actual bytes, which means
      // knowing the element type; op_repr records it alongside the value.
      std::string value, dtype;
      if (parse_splat_constant(repr, value, dtype)) {
        int64 bytes = tensor->get_sizes()[0];
        int64 width = dtype_width(dtype);
        assert(bytes % width == 0 && "constant size is not a whole number of "
                                     "elements");
        std::ostringstream v;
        v << "jax.lax.bitcast_convert_type(jnp.full((" << bytes / width
          << ",), " << value << ", " << jnp_dtype(dtype) << "), jnp.uint8)"
          << ".reshape(-1)";
        constant_info.emplace_back(MetaDataInfo(tensor->get_offsets()[0]->Min(),
                                                tensor->get_sizes(), "jnp.uint8",
                                                v.str()));
        continue;
      }

      // Anything else would have to be emitted as bytes we cannot derive.
      // Refuse rather than substitute a placeholder: a wrong constant is a
      // silently wrong kernel.
      std::cerr << "Error: cannot emit constant tensor " << tensor->get_name()
                << " with value " << repr << std::endl;
      assert(false && "unsupported constant tensor value");
    }
  }

  this->max_hbm_size = 0;
  for (const auto &pair : tensor_map) {
    auto *tensor = pair.second;
    if (tensor->get_storage()->get_name() == "HBM") {
      int64 offset = tensor->get_offsets()[0]->Min();
      int64 size = tensor->get_sizes()[0];
      if (offset + size > this->max_hbm_size)
        this->max_hbm_size = offset + size;
    }
  }
}

bool assembly_dump(char *outpath,
                   const std::vector<Instruction *> &instructions,
                   const MetaData &metadata) {
  std::string indent1 = "    ";
  std::string indent2 = indent1 + indent1;
  std::string indent3 = indent2 + indent1;
  std::string indent4 = indent3 + indent1;

  // Process hlo_name and pii_number from outpath.
  // Accept any path with a filename stem; use parent directory as hlo_name
  // when available, otherwise fall back to metadata.module_name.
  std::string path_str(outpath);
  size_t lslash = path_str.find_last_of('/');
  size_t fname_start = (lslash == std::string::npos) ? 0 : lslash + 1;
  if (fname_start >= path_str.size()) {
    std::cerr
        << "Warning: output path " << outpath
        << " has no filename component. Defaulting to no solution."
        << std::endl;
    return false;
  }

  size_t ldot = path_str.find_last_of('.');
  size_t stem_end = path_str.size();
  if (ldot != std::string::npos && ldot > fname_start) {
    stem_end = ldot;
  }

  if (stem_end <= fname_start) {
    std::cerr
        << "Warning: output path " << outpath
        << " does not contain a valid filename stem. "
           "Defaulting to no solution."
        << std::endl;
    return false;
  }

  std::string pii_number = path_str.substr(fname_start, stem_end - fname_start);
  std::string hlo_name = metadata.module_name;
  if (lslash != std::string::npos && lslash > 0) {
    size_t l2slash = path_str.find_last_of('/', lslash - 1);
    size_t parent_start = (l2slash == std::string::npos) ? 0 : l2slash + 1;
    if (lslash > parent_start) {
      hlo_name = path_str.substr(parent_start, lslash - parent_start);
    }
  }

  std::ofstream outfile(outpath);
  if (!outfile) {
    std::cerr << "Warning: failed to open output file: " << outpath
              << "; defaulting to no solution." << std::endl;
    return false;
  }

  // Process kernel function name
  std::string kernel_name = metadata.module_name;

  // HEADER
  outfile << "# Input file: " << hlo_name << ".hlo" << std::endl;
  outfile << "# Kernel name: " << kernel_name << std::endl;
  outfile << "# PII number: " << pii_number << std::endl;
  outfile << "# Do not edit!" << std::endl << std::endl;

  outfile << "import jax" << std::endl;
  outfile << "import jax.numpy as jnp" << std::endl << std::endl << std::endl;

  // Kernel function metadata
  outfile << "def " << kernel_name << "(kernel, api):" << std::endl;

  outfile << indent1 << "@kernel(hbm=" << metadata.max_hbm_size << ","
          << std::endl;

  outfile << indent3 << "input=[" << std::endl;
  for (const auto &info : metadata.input_info) {
    outfile << indent4 << info.str() << "," << std::endl;
  }
  outfile << indent3 << "]," << std::endl;

  if (!metadata.constant_info.empty()) {
    outfile << indent3 << "constant=[" << std::endl;
    for (const auto &info : metadata.constant_info) {
      outfile << indent4 << info.str() << "," << std::endl;
    }
    outfile << indent3 << "]," << std::endl;
  } else {
    outfile << indent3 << "constant=[]," << std::endl;
  }

  outfile << indent3 << "output=[" << std::endl;
  for (const auto &info : metadata.output_info) {
    outfile << indent4 << info.str() << "," << std::endl;
  }
  outfile << indent3 << "]" << std::endl;

  outfile << indent3 << ")" << std::endl;
  outfile << indent1 << "def " << kernel_name << "_" << "():" << std::endl;

  // Assembly code
  for (auto *instr : instructions) {
    outfile << indent2 << "api." << instr->str() << std::endl;
  }

  outfile << std::endl;
  outfile << indent1 << "return " << kernel_name << "_" << std::endl;

  outfile.close();
  return true;
}

} // namespace emitter
