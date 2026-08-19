#include "llm/qwen3_tokenizer.h"

#include <Python.h>

#include <mutex>
#include <stdexcept>

namespace llm {
namespace {

std::once_flag python_once;

void ensure_python() {
  std::call_once(python_once, [] {
    Py_Initialize();
    if (!Py_IsInitialized()) throw std::runtime_error("Qwen3Tokenizer: failed to initialize Python runtime");
    PyEval_SaveThread();
  });
}

std::string python_error(const std::string& context) {
  if (!PyErr_Occurred()) return context;
  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);
  PyObject* text = value ? PyObject_Str(value) : nullptr;
  const char* message = text ? PyUnicode_AsUTF8(text) : nullptr;
  std::string result = context;
  if (message) result += ": " + std::string(message);
  Py_XDECREF(text); Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(traceback);
  return result;
}

class GIL {
 public:
  GIL() : state_(PyGILState_Ensure()) {}
  ~GIL() { PyGILState_Release(state_); }
 private:
  PyGILState_STATE state_;
};

}  // namespace

Qwen3Tokenizer::Qwen3Tokenizer(const std::string& path) {
  if (path.empty()) throw std::invalid_argument("Qwen3Tokenizer: tokenizer JSON path is empty");
  ensure_python();
  GIL gil;
  PyObject* module = PyImport_ImportModule("tokenizers");
  if (!module) throw std::runtime_error(python_error("Qwen3Tokenizer: cannot import tokenizers"));
  PyObject* cls = PyObject_GetAttrString(module, "Tokenizer");
  PyObject* from_file = cls ? PyObject_GetAttrString(cls, "from_file") : nullptr;
  PyObject* arg = PyUnicode_FromString(path.c_str());
  PyObject* object = from_file && arg ? PyObject_CallFunctionObjArgs(from_file, arg, nullptr) : nullptr;
  Py_XDECREF(arg); Py_XDECREF(from_file); Py_XDECREF(cls); Py_DECREF(module);
  if (!object) throw std::runtime_error(python_error("Qwen3Tokenizer: failed to load tokenizer JSON '" + path + "'"));
  tokenizer_ = object;

  PyObject* size = PyObject_CallMethod(static_cast<PyObject*>(tokenizer_), "get_vocab_size", nullptr);
  if (!size || !PyLong_Check(size)) {
    Py_XDECREF(size); Py_DECREF(static_cast<PyObject*>(tokenizer_)); tokenizer_ = nullptr;
    throw std::runtime_error(python_error("Qwen3Tokenizer: failed to read vocabulary size"));
  }
  long value = PyLong_AsLong(size); Py_DECREF(size);
  if (value <= 0 || value > INT32_MAX) {
    Py_DECREF(static_cast<PyObject*>(tokenizer_)); tokenizer_ = nullptr;
    throw std::runtime_error("Qwen3Tokenizer: invalid vocabulary size");
  }
  vocab_size_ = static_cast<int32_t>(value);
}

Qwen3Tokenizer::~Qwen3Tokenizer() {
  if (!tokenizer_ || !Py_IsInitialized()) return;
  GIL gil;
  Py_DECREF(static_cast<PyObject*>(tokenizer_));
  tokenizer_ = nullptr;
}

std::vector<int32_t> Qwen3Tokenizer::encode(const std::string& text) const {
  if (!tokenizer_) throw std::runtime_error("Qwen3Tokenizer::encode: tokenizer is not initialized");
  GIL gil;
  PyObject* input = PyUnicode_FromStringAndSize(text.data(), static_cast<Py_ssize_t>(text.size()));
  PyObject* encoding = input ? PyObject_CallMethod(static_cast<PyObject*>(tokenizer_), "encode", "O", input) : nullptr;
  Py_XDECREF(input);
  if (!encoding) throw std::runtime_error(python_error("Qwen3Tokenizer::encode failed"));
  PyObject* ids = PyObject_GetAttrString(encoding, "ids");
  if (!ids || !PySequence_Check(ids)) {
    Py_XDECREF(ids); Py_DECREF(encoding);
    throw std::runtime_error(python_error("Qwen3Tokenizer::encode returned invalid ids"));
  }
  const Py_ssize_t n = PySequence_Size(ids);
  std::vector<int32_t> result; result.reserve(static_cast<size_t>(n));
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PySequence_GetItem(ids, i);
    long id = item ? PyLong_AsLong(item) : -1; Py_XDECREF(item);
    if (PyErr_Occurred() || id < 0 || id >= vocab_size_) {
      Py_DECREF(ids); Py_DECREF(encoding);
      throw std::runtime_error(python_error("Qwen3Tokenizer::encode produced invalid token id"));
    }
    result.push_back(static_cast<int32_t>(id));
  }
  Py_DECREF(ids); Py_DECREF(encoding);
  return result;
}

std::string Qwen3Tokenizer::decode(const std::vector<int32_t>& ids) const {
  if (!tokenizer_) throw std::runtime_error("Qwen3Tokenizer::decode: tokenizer is not initialized");
  GIL gil;
  PyObject* list = PyList_New(static_cast<Py_ssize_t>(ids.size()));
  if (!list) throw std::runtime_error(python_error("Qwen3Tokenizer::decode allocation failed"));
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0 || ids[i] >= vocab_size_) {
      Py_DECREF(list);
      throw std::out_of_range("Qwen3Tokenizer::decode: token id " + std::to_string(ids[i]) +
                              " outside [0," + std::to_string(vocab_size_) + ")");
    }
    PyObject* value = PyLong_FromLong(ids[i]);
    if (!value) { Py_DECREF(list); throw std::runtime_error(python_error("Qwen3Tokenizer::decode id allocation failed")); }
    PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), value);
  }
  PyObject* decoded = PyObject_CallMethod(static_cast<PyObject*>(tokenizer_), "decode", "O", list);
  Py_DECREF(list);
  if (!decoded) throw std::runtime_error(python_error("Qwen3Tokenizer::decode failed"));
  Py_ssize_t length = 0; const char* text = PyUnicode_AsUTF8AndSize(decoded, &length);
  if (!text) { Py_DECREF(decoded); throw std::runtime_error(python_error("Qwen3Tokenizer::decode returned non-UTF-8 text")); }
  std::string result(text, static_cast<size_t>(length)); Py_DECREF(decoded); return result;
}

std::optional<int32_t> Qwen3Tokenizer::eos_token_id() const {
  if (!tokenizer_) throw std::runtime_error("Qwen3Tokenizer::eos_token_id: tokenizer is not initialized");
  GIL gil;
  PyObject* token_to_id = PyObject_GetAttrString(static_cast<PyObject*>(tokenizer_), "token_to_id");
  PyObject* token = PyUnicode_FromString("<|im_end|>");
  PyObject* id = token_to_id && token ? PyObject_CallFunctionObjArgs(token_to_id, token, nullptr) : nullptr;
  Py_XDECREF(token); Py_XDECREF(token_to_id);
  if (!id) { PyErr_Clear(); return std::nullopt; }
  if (id == Py_None) { Py_DECREF(id); return std::nullopt; }
  long value = PyLong_AsLong(id); Py_DECREF(id);
  if (PyErr_Occurred() || value < 0 || value >= vocab_size_) return std::nullopt;
  return static_cast<int32_t>(value);
}

}  // namespace llm
