#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/Basic/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/CharInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Format.h"
#include <memory>
#include <utility>
#include <sstream>

namespace ultima{

namespace detail{

template<std::size_t N>struct priority : priority<N-1>{};
template<>struct priority<0>{};

struct hasPrintTemplateArgumentList{
  template<typename T, typename... Args>
  static auto check(priority<1>, Args&&...) -> decltype(T::PrintTemplateArgumentList(std::declval<Args>()...), std::true_type{});
  template<typename T, typename... Args>
  static std::false_type check(priority<0>, Args&&...);
};

template<typename T, typename... Args>
static inline auto printTemplateArgumentList(Args&&... args)->decltype(T::PrintTemplateArgumentList(std::forward<Args>(args)...)){
  return T::PrintTemplateArgumentList(std::forward<Args>(args)...);
}

}

}

namespace clang{

template<typename... Args, typename std::enable_if<decltype(ultima::detail::hasPrintTemplateArgumentList::check<clang::TemplateSpecializationType>(ultima::detail::priority<1>{}, std::declval<Args>()...))::value, std::nullptr_t>::type = nullptr>
static inline auto printTemplateArgumentList(Args&&... args)->decltype(ultima::detail::printTemplateArgumentList<clang::TemplateSpecializationType>(std::forward<Args>(args)...)){
  return ultima::detail::printTemplateArgumentList<clang::TemplateSpecializationType>(std::forward<Args>(args)...);
}

}

namespace ultima{

struct ostreams{
  std::vector<llvm::raw_ostream*> oss;
  ostreams(llvm::raw_ostream& os):oss{&os}{}
  template<typename T>
  llvm::raw_ostream& operator<<(T&& rhs){return (*oss.back()) << rhs;}
  operator llvm::raw_ostream&(){return *oss.back();}
  void push(llvm::raw_ostream& os){oss.emplace_back(&os);}
  void pop(){oss.pop_back();}
  struct auto_popper{
    ostreams* oss;
    auto_popper(ostreams& oss, llvm::raw_ostream& os):oss{&oss}{oss.push(os);}
    auto_popper(auto_popper&& other):oss{other.oss}{other.oss = nullptr;}
    ~auto_popper(){if(oss)oss->pop();}
  };
  auto_popper scoped_push(llvm::raw_ostream& os){return {*this, os};}
};

namespace detail{

template<typename PPCallbacks>
struct pp_callbacks : PPCallbacks{
  virtual void InclusionDirective(
    clang::SourceLocation,
    const clang::Token&,
    llvm::StringRef,
    bool,
    clang::CharSourceRange,
    const clang::FileEntry*,
    llvm::StringRef,
    llvm::StringRef,
    const clang::Module*
  ) = 0;
  void InclusionDirective(
    clang::SourceLocation hash_loc,
    const clang::Token& include_tok,
    llvm::StringRef filename,
    bool is_angled,
    clang::CharSourceRange filename_range,
    const clang::FileEntry* file,
    llvm::StringRef search_path,
    llvm::StringRef relative_path,
    const clang::Module* imported,
    clang::SrcMgr::CharacteristicKind
  )override{
    this->InclusionDirective(hash_loc, include_tok, filename, is_angled, filename_range, file, search_path, relative_path, imported);
  }
};

template<typename T>
struct inclusion_directive_is_callable_with_characteristic_kind{
  template<typename U>
  static std::true_type impl(
    decltype(std::declval<U>().InclusionDirective(
      std::declval<clang::SourceLocation>(),
      std::declval<const clang::Token&>(),
      std::declval<llvm::StringRef>(),
      std::declval<bool>(),
      std::declval<clang::CharSourceRange>(),
      std::declval<const clang::FileEntry*>(),
      std::declval<llvm::StringRef>(),
      std::declval<llvm::StringRef>(),
      std::declval<const clang::Module*>(),
      std::declval<clang::SrcMgr::CharacteristicKind>()))*
  );
  template<typename U>
  static std::false_type impl(
    decltype(std::declval<U>().InclusionDirective(
      std::declval<clang::SourceLocation>(),
      std::declval<const clang::Token&>(),
      std::declval<llvm::StringRef>(),
      std::declval<bool>(),
      std::declval<clang::CharSourceRange>(),
      std::declval<const clang::FileEntry*>(),
      std::declval<llvm::StringRef>(),
      std::declval<llvm::StringRef>(),
      std::declval<const clang::Module*>()))*
  );
  using type = decltype(impl<T>(nullptr));
};

}

template<typename PPCallbacks>
using pp_callbacks = typename std::conditional<
  detail::inclusion_directive_is_callable_with_characteristic_kind<PPCallbacks>::type::value,
  detail::pp_callbacks<PPCallbacks>,
  PPCallbacks
>::type;

class preprocessor : public pp_callbacks<clang::PPCallbacks>{
  void output(char start, llvm::StringRef filename, char end){
    llvm::outs() << start << filename << end << '\n';
  }
 public:
  constexpr preprocessor() = default;
  void InclusionDirective(
    clang::SourceLocation,
    const clang::Token&,
    llvm::StringRef filename,
    bool is_angled,
    clang::CharSourceRange,
    const clang::FileEntry*,
    llvm::StringRef,
    llvm::StringRef,
    const clang::Module*
  )override{
    if(filename == "cuda_stub.hpp" || filename == "cl_stub.hpp")
      return;
    llvm::outs() << "#include";
    std::string fn;
    static constexpr const char cupy_dir[] = "cupy/";
    static constexpr std::size_t cupy_dir_length = sizeof(cupy_dir)-1;
    static constexpr const char hpp_ext[] = ".hpp";
    static constexpr std::size_t hpp_ext_length = sizeof(hpp_ext)-1;
    if(filename.startswith(cupy_dir))
      if(filename.endswith(hpp_ext))
        fn = "clpy/" + filename.substr(cupy_dir_length, filename.size()-cupy_dir_length-hpp_ext_length).str() + ".clh";
      else
        fn = "clpy/" + filename.substr(cupy_dir_length, filename.size()-cupy_dir_length).str();
    else
      fn = filename;
    output(
      is_angled ? '<'             : '"',
      fn,
      is_angled ? '>'             : '"'
    );
  }
};

namespace detail{

template<typename PredefinedExpr>
static inline decltype(PredefinedExpr::getIdentTypeName(std::declval<PredefinedExpr>().getIdentType())) getIdentTypeName(PredefinedExpr* pe){
  return PredefinedExpr::getIdentTypeName(pe->getIdentType());
}

template<typename PredefinedExpr>
static inline decltype(PredefinedExpr::getIdentKindName(std::declval<PredefinedExpr>().getIdentKind())) getIdentTypeName(PredefinedExpr* pe){
  return PredefinedExpr::getIdentKindName(pe->getIdentKind());
}

template<typename GenericSelectionExpr>
class generic_selection_expr_index{
  GenericSelectionExpr* t;
 public:
  class iterator{
    class accessor{
      GenericSelectionExpr* t;
      unsigned index;
     public:
      accessor() = default;
      accessor(GenericSelectionExpr* ptr, unsigned ind):t{ptr}, index{ind}{}
      accessor(const accessor&) = default;
      accessor(accessor&&) = default;
      clang::QualType get_type()const{
        return t->getAssocType(index);
      }
      clang::Expr* get_expr()const{
        return t->getAssocExpr(index);
      }
      friend class iterator;
    }t;
   public:
    iterator() = default;
    iterator(GenericSelectionExpr* ptr, unsigned ind):t{ptr, ind}{}
    iterator(const iterator&) = default;
    iterator(iterator&&) = default;
    iterator& operator++()noexcept{
      ++t.index;
      return *this;
    }
    iterator operator++(int){
      iterator it = *this;
      ++t.index;
      return it;
    }
    bool operator==(const iterator& other)const{
      return t.index == other.t.index;
    }
    bool operator!=(const iterator& other)const{
      return !(*this == other);
    }
    accessor& operator*()noexcept{
      return t;
    }
    const accessor& operator*()const noexcept{
      return t;
    }
    accessor* operator->()noexcept{
      return &t;
    }
    const accessor* operator->()const noexcept{
      return &t;
    }
  };
  constexpr generic_selection_expr_index(GenericSelectionExpr* ptr):t{ptr}{}
  iterator begin()const{return iterator{t, 0u};}
  iterator end()const{return iterator{t, t->getNumAssocs()};}
};

template<typename GenericSelectionExpr>
class generic_selection_expr_iterator{
  GenericSelectionExpr* t;
 public:
  class iterator{
    class accessor{
      typename GenericSelectionExpr::AssociationIterator t;
     public:
      accessor() = default;
      accessor(const typename GenericSelectionExpr::AssociationIterator& it):t{it}{}
      accessor(const accessor&) = default;
      accessor(accessor&&) = default;
      clang::QualType get_type()const{
        return t->getType();
      }
      clang::Expr* get_expr()const{
        return t->getAssociationExpr();
      }
      friend class iterator;
    }t;
   public:
    iterator() = default;
    iterator(typename GenericSelectionExpr::AssociationIterator itr):t{itr}{}
    iterator(const iterator&) = default;
    iterator(iterator&&) = default;
    iterator& operator++()noexcept{
      ++t.t;
      return *this;
    }
    iterator operator++(int){
      iterator it = *this;
      ++t.t;
      return it;
    }
    bool operator==(const iterator& other)const{
      return t.t == other.t.t;
    }
    bool operator!=(const iterator& other)const{
      return !(*this == other);
    }
    accessor& operator*()noexcept{
      return t;
    }
    const accessor& operator*()const noexcept{
      return t;
    }
    accessor* operator->()noexcept{
      return &t;
    }
    const accessor* operator->()const noexcept{
      return &t;
    }
  };
  constexpr generic_selection_expr_iterator(GenericSelectionExpr* ptr):t{ptr}{}
  iterator begin()const{return iterator{t->associations().begin()};}
  iterator end()const{return iterator{t->associations().end()};}
};

template<typename GenericSelectionExpr, typename std::enable_if<std::is_same<clang::QualType, decltype(std::declval<GenericSelectionExpr>().getAssocType(0u))>::value, std::nullptr_t>::type = nullptr>
static constexpr generic_selection_expr_index<GenericSelectionExpr> generic_selection_expr_wrapper(GenericSelectionExpr* t){
  return {t};
}
template<typename GenericSelectionExpr, typename std::enable_if<std::is_same<clang::QualType, decltype(std::declval<GenericSelectionExpr>().associations().begin()->getType())>::value, std::nullptr_t>::type = nullptr>
static constexpr generic_selection_expr_iterator<GenericSelectionExpr> generic_selection_expr_wrapper(GenericSelectionExpr* t){
  return {t};
}

template<typename CXXNewExpr, typename std::enable_if<std::is_same<clang::Expr*, decltype(std::declval<CXXNewExpr>().getArraySize())>::value, std::nullptr_t>::type = nullptr>
static inline clang::Expr* get_array_size(CXXNewExpr* node){
   return node->getArraySize();
}

template<typename CXXNewExpr, typename std::enable_if<std::is_same<clang::Optional<clang::Expr*>, decltype(std::declval<CXXNewExpr>().getArraySize())>::value, std::nullptr_t>::type = nullptr>
static inline clang::Expr* get_array_size(CXXNewExpr* node){
   auto e = node->getArraySize();
   return e ? *e : nullptr;
}

}

class decl_visitor;

template<typename T>
static inline void printGroup(clang::DeclVisitor<T>& t, clang::Decl** b, std::size_t s){
  static_cast<T&>(t).printGroup(b, s);
}
template<typename T, typename U>
static inline void prettyPrintAttributes(clang::DeclVisitor<T>& t, U* u){
  static_cast<T&>(t).prettyPrintAttributes(u);
}

struct function_special_argument_info{
  std::string name;
  std::string type;
  enum{
    raw,
    ind,
    cindex
  }arg_flag;
  int ndim;
  bool is_input;
};

class stmt_visitor : public clang::StmtVisitor<stmt_visitor> {
  ostreams& os;
  unsigned& IndentLevel;
  clang::PrintingPolicy& Policy;
  clang::DeclVisitor<decl_visitor>& dv;
  const std::vector<std::vector<function_special_argument_info>>& func_arg_info;
  const std::unordered_map<clang::FunctionDecl*, std::string>& func_name;
  std::vector<std::string>& delayed_outputs;
public:
  static bool has_annotation(clang::Decl* decl, llvm::StringRef s){
    if(decl->hasAttrs())
      for(auto&& x : decl->getAttrs())
        if(auto a = clang::dyn_cast<clang::AnnotateAttr>(x))
          if(s == a->getAnnotation())
            return true;
    return false;
  }
  stmt_visitor(ostreams& os,
              clang::PrintingPolicy &Policy,
              unsigned& Indentation, clang::DeclVisitor<decl_visitor>& dv,
              const std::vector<std::vector<function_special_argument_info>>& func_arg_info,
              const std::unordered_map<clang::FunctionDecl*, std::string>& func_name,
              std::vector<std::string>& delayed_outputs)
    : os(os), IndentLevel(Indentation), Policy(Policy), dv{dv}, func_arg_info{func_arg_info}, func_name{func_name}, delayed_outputs{delayed_outputs} {}

  void PrintStmt(clang::Stmt *S) {
    PrintStmt(S, Policy.Indentation);
  }

  void PrintStmt(clang::Stmt *S, int SubIndent) {
    IndentLevel += SubIndent;
    if (S && clang::isa<clang::Expr>(S)) {
      // If this is an expr used in a stmt context, indent and newline it.
      Indent();
      Visit(S);
      os << ";\n";
    } else if (S) {
      Visit(S);
    } else {
      Indent() << "/*<<<NULL STATEMENT>>>*/\n";
    }
    IndentLevel -= SubIndent;
  }

  void PrintExpr(clang::Expr *E) {
    if (E)
      Visit(E);
    else
      os << "/*<null expr>*/";
  }

  llvm::raw_ostream &Indent(int Delta = 0) {
    for (int i = 0, e = IndentLevel+Delta; i < e; ++i)
      os << "  ";
    return os;
  }

  void VisitStmt(clang::Stmt*) LLVM_ATTRIBUTE_UNUSED {
    Indent() << "<<unknown stmt type>>\n";
  }
  void VisitExpr(clang::Expr*) LLVM_ATTRIBUTE_UNUSED {
    os << "<<unknown expr type>>";
  }

  /// PrintRawCompoundStmt - Print a compound stmt without indenting the {, and
  /// with no newline after the }.
  void PrintRawCompoundStmt(clang::CompoundStmt *Node) {
    os << "{\n";
    for (auto *I : Node->body())
      PrintStmt(I);

    Indent() << '}';
  }

  void PrintRawDecl(clang::Decl *D) {
    dv.Visit(D);
  }

  void PrintRawDeclStmt(const clang::DeclStmt *S) {
    llvm::SmallVector<clang::Decl*, 2> Decls(S->decls());
    printGroup(dv, Decls.data(), Decls.size());
  }

  void VisitNullStmt(clang::NullStmt*) {
    Indent() << ";\n";
  }

  void VisitDeclStmt(clang::DeclStmt *Node) {
    Indent();
    PrintRawDeclStmt(Node);
    os << ';';
    if(!delayed_outputs.back().empty()){
      os << delayed_outputs.back();
      delayed_outputs.pop_back();
    }
    os << '\n';
  }

  void VisitCompoundStmt(clang::CompoundStmt *Node) {
    Indent();
    PrintRawCompoundStmt(Node);
    os << "\n";
  }

  void VisitCaseStmt(clang::CaseStmt *Node) {
    Indent(-1) << "case ";
    PrintExpr(Node->getLHS());
    if (Node->getRHS()) {
      os << " ... ";
      PrintExpr(Node->getRHS());
    }
    os << ":\n";

    PrintStmt(Node->getSubStmt(), 0);
  }

  void VisitDefaultStmt(clang::DefaultStmt *Node) {
    Indent(-1) << "default:\n";
    PrintStmt(Node->getSubStmt(), 0);
  }

  void VisitLabelStmt(clang::LabelStmt *Node) {
    Indent(-1) << Node->getName() << ":\n";
    PrintStmt(Node->getSubStmt(), 0);
  }

  void VisitAttributedStmt(clang::AttributedStmt *Node) {
    prettyPrintAttributes(dv, Node);
    PrintStmt(Node->getSubStmt(), 0);
  }

  void PrintRawIfStmt(clang::IfStmt *If) {
    os << "if (";
    if (const auto *DS = If->getConditionVariableDeclStmt())
      PrintRawDeclStmt(DS);
    else
      PrintExpr(If->getCond());
    os << ')';

    if (auto *CS = clang::dyn_cast<clang::CompoundStmt>(If->getThen())) {
      os << ' ';
      PrintRawCompoundStmt(CS);
      os << (If->getElse() ? ' ' : '\n');
    } else {
      os << '\n';
      PrintStmt(If->getThen());
      if (If->getElse()) Indent();
    }

    if (auto *Else = If->getElse()) {
      os << "else";

      if (auto *CS = clang::dyn_cast<clang::CompoundStmt>(Else)) {
        os << ' ';
        PrintRawCompoundStmt(CS);
        os << '\n';
      } else if (auto *ElseIf = clang::dyn_cast<clang::IfStmt>(Else)) {
        os << ' ';
        PrintRawIfStmt(ElseIf);
      } else {
        os << '\n';
        PrintStmt(If->getElse());
      }
    }
  }

  void VisitIfStmt(clang::IfStmt *If) {
    Indent();
    PrintRawIfStmt(If);
  }

  void VisitSwitchStmt(clang::SwitchStmt *Node) {
    Indent() << "switch (";
    if (const auto *DS = Node->getConditionVariableDeclStmt())
      PrintRawDeclStmt(DS);
    else
      PrintExpr(Node->getCond());
    os << ')';

    // Pretty print compoundstmt bodies (very common).
    if (auto *CS = clang::dyn_cast<clang::CompoundStmt>(Node->getBody())) {
      os << ' ';
      PrintRawCompoundStmt(CS);
      os << '\n';
    } else {
      os << '\n';
      PrintStmt(Node->getBody());
    }
  }

  void VisitWhileStmt(clang::WhileStmt *Node) {
    Indent() << "while (";
    if (const auto *DS = Node->getConditionVariableDeclStmt())
      PrintRawDeclStmt(DS);
    else
      PrintExpr(Node->getCond());
    os << ")\n";
    PrintStmt(Node->getBody());
  }

  void VisitDoStmt(clang::DoStmt *Node) {
    Indent() << "do ";
    if (auto *CS = clang::dyn_cast<clang::CompoundStmt>(Node->getBody())) {
      PrintRawCompoundStmt(CS);
      os << ' ';
    } else {
      os << "\n";
      PrintStmt(Node->getBody());
      Indent();
    }

    os << "while (";
    PrintExpr(Node->getCond());
    os << ");\n";
  }

  void VisitForStmt(clang::ForStmt *Node) {
    Indent() << "for (";
    if (Node->getInit()) {
      if (auto *DS = clang::dyn_cast<clang::DeclStmt>(Node->getInit()))
        PrintRawDeclStmt(DS);
      else
        PrintExpr(clang::cast<clang::Expr>(Node->getInit()));
    }
    os << ';';
    if (Node->getCond()) {
      os << ' ';
      PrintExpr(Node->getCond());
    }
    os << ';';
    if (Node->getInc()) {
      os << ' ';
      PrintExpr(Node->getInc());
    }
    os << ") ";

    if (auto *CS = clang::dyn_cast<clang::CompoundStmt>(Node->getBody())) {
      PrintRawCompoundStmt(CS);
      os << '\n';
    } else {
      os << '\n';
      PrintStmt(Node->getBody());
    }
  }

  void VisitCXXForRangeStmt(clang::CXXForRangeStmt *Node) {
    Indent() << "for (";
    auto backup = Policy;
    Policy.SuppressInitializers = true;
    dv.Visit(Node->getLoopVariable());
    Policy = backup;
    os << " : ";
    PrintExpr(Node->getRangeInit());
    os << ") {\n";
    PrintStmt(Node->getBody());
    Indent() << '}';
    if (Policy.IncludeNewlines) os << '\n';
  }

  void VisitMSDependentExistsStmt(clang::MSDependentExistsStmt *Node) {
    Indent();
    if (Node->isIfExists())
      os << "__if_exists (";
    else
      os << "__if_not_exists (";
    
    if (auto *Qualifier
          = Node->getQualifierLoc().getNestedNameSpecifier())
      Qualifier->print(os, Policy);
    
    os << Node->getNameInfo() << ") ";
    
    PrintRawCompoundStmt(Node->getSubStmt());
  }

  void VisitGotoStmt(clang::GotoStmt *Node) {
    Indent() << "goto " << Node->getLabel()->getName() << ';';
    if (Policy.IncludeNewlines) os << '\n';
  }

  void VisitIndirectGotoStmt(clang::IndirectGotoStmt *Node) {
    Indent() << "goto *";
    PrintExpr(Node->getTarget());
    os << ';';
    if (Policy.IncludeNewlines) os << '\n';
  }

  void VisitContinueStmt(clang::ContinueStmt*) {
    Indent() << "continue;";
    if (Policy.IncludeNewlines) os << '\n';
  }

  void VisitBreakStmt(clang::BreakStmt*) {
    Indent() << "break;";
    if (Policy.IncludeNewlines) os << '\n';
  }


  void VisitReturnStmt(clang::ReturnStmt *Node) {
    Indent() << "return";
    if (Node->getRetValue()) {
      os << ' ';
      PrintExpr(Node->getRetValue());
    }
    os << ';';
    if (Policy.IncludeNewlines) os << '\n';
  }


  void VisitGCCAsmStmt(clang::GCCAsmStmt *Node) {
    Indent() << "asm ";

    if (Node->isVolatile())
      os << "volatile ";

    os << '(';
    VisitStringLiteral(Node->getAsmString());

    // Outputs
    if (Node->getNumOutputs() != 0 || Node->getNumInputs() != 0 ||
        Node->getNumClobbers() != 0)
      os << " : ";

    for (unsigned i = 0, e = Node->getNumOutputs(); i != e; ++i) {
      if (i != 0)
        os << ", ";

      if (!Node->getOutputName(i).empty()) {
        os << '[';
        os << Node->getOutputName(i);
        os << "] ";
      }

      VisitStringLiteral(Node->getOutputConstraintLiteral(i));
      os << " (";
      Visit(Node->getOutputExpr(i));
      os << ')';
    }

    // Inputs
    if (Node->getNumInputs() != 0 || Node->getNumClobbers() != 0)
      os << " : ";

    for (unsigned i = 0, e = Node->getNumInputs(); i != e; ++i) {
      if (i != 0)
        os << ", ";

      if (!Node->getInputName(i).empty()) {
        os << '[';
        os << Node->getInputName(i);
        os << "] ";
      }

      VisitStringLiteral(Node->getInputConstraintLiteral(i));
      os << " (";
      Visit(Node->getInputExpr(i));
      os << ')';
    }

    // Clobbers
    if (Node->getNumClobbers() != 0)
      os << " : ";

    for (unsigned i = 0, e = Node->getNumClobbers(); i != e; ++i) {
      if (i != 0)
        os << ", ";

      VisitStringLiteral(Node->getClobberStringLiteral(i));
    }

    os << ");";
    if (Policy.IncludeNewlines) os << '\n';
  }

  void VisitMSAsmStmt(clang::MSAsmStmt *Node) {
    // FIXME: Implement MS style inline asm statement printer.
    Indent() << "__asm ";
    if (Node->hasBraces())
      os << "{\n";
    os << Node->getAsmString() << '\n';
    if (Node->hasBraces())
      Indent() << "}\n";
  }

  void VisitCapturedStmt(clang::CapturedStmt *Node) {
    PrintStmt(Node->getCapturedDecl()->getBody());
  }

  void PrintRawCXXCatchStmt(clang::CXXCatchStmt *Node) {
    os << "catch (";
    if (clang::Decl *ExDecl = Node->getExceptionDecl())
      PrintRawDecl(ExDecl);
    else
      os << "...";
    os << ") ";
    PrintRawCompoundStmt(clang::cast<clang::CompoundStmt>(Node->getHandlerBlock()));
  }

  void VisitCXXCatchStmt(clang::CXXCatchStmt *Node) {
    Indent();
    PrintRawCXXCatchStmt(Node);
    os << '\n';
  }

  void VisitCXXTryStmt(clang::CXXTryStmt *Node) {
    Indent() << "try ";
    PrintRawCompoundStmt(Node->getTryBlock());
    for (unsigned i = 0, e = Node->getNumHandlers(); i < e; ++i) {
      os << ' ';
      PrintRawCXXCatchStmt(Node->getHandler(i));
    }
    os << '\n';
  }

  void VisitDeclRefExpr(clang::DeclRefExpr *Node) {
    if (auto *Qualifier = Node->getQualifier())
      Qualifier->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }

  void VisitDependentScopeDeclRefExpr(clang::DependentScopeDeclRefExpr *Node) {
    if (auto *Qualifier = Node->getQualifier())
      Qualifier->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }

  void VisitUnresolvedLookupExpr(clang::UnresolvedLookupExpr *Node) {
    if (Node->getQualifier())
      Node->getQualifier()->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }

  void VisitPredefinedExpr(clang::PredefinedExpr *Node) {
    os << detail::getIdentTypeName(Node);
  }

  void VisitCharacterLiteral(clang::CharacterLiteral *Node) {
    Node->printPretty(os, nullptr, Policy);
  }

  void VisitIntegerLiteral(clang::IntegerLiteral *Node) {
    Node->printPretty(os, nullptr, Policy);
  }

  static void PrintFloatingLiteral(llvm::raw_ostream &os, clang::FloatingLiteral *Node,
                                   bool PrintSuffix) {
    llvm::SmallString<16> Str;
    Node->getValue().toString(Str);
    os << Str;
    if (Str.find_first_not_of("-0123456789") == llvm::StringRef::npos)
      os << '.'; // Trailing dot in order to separate from ints.

    if (!PrintSuffix)
      return;

    // Emit suffixes.  Float literals are always a builtin float type.
    switch (Node->getType()->getAs<clang::BuiltinType>()->getKind()) {
    default: llvm_unreachable("Unexpected type for float literal!");
    case clang::BuiltinType::Half:       break; // FIXME: suffix?
    case clang::BuiltinType::Double:     break; // no suffix.
    case clang::BuiltinType::Float:      os << 'F'; break;
    case clang::BuiltinType::LongDouble: os << 'L'; break;
    case clang::BuiltinType::Float128:   os << 'Q'; break;
    }
  }

  void VisitFloatingLiteral(clang::FloatingLiteral *Node) {
    PrintFloatingLiteral(os, Node, /*PrintSuffix=*/true);
  }

  void VisitImaginaryLiteral(clang::ImaginaryLiteral *Node) {
    PrintExpr(Node->getSubExpr());
    os << 'i';
  }

  void VisitStringLiteral(clang::StringLiteral *Str) {
    Str->outputString(os);
  }
  void VisitParenExpr(clang::ParenExpr *Node) {
    os << '(';
    PrintExpr(Node->getSubExpr());
    os << ')';
  }
  void VisitUnaryOperator(clang::UnaryOperator *Node) {
    if (!Node->isPostfix()) {
      os << clang::UnaryOperator::getOpcodeStr(Node->getOpcode());

      // Print a space if this is an "identifier operator" like __real, or if
      // it might be concatenated incorrectly like '+'.
      switch (Node->getOpcode()) {
      default: break;
      case clang::UO_Real:
      case clang::UO_Imag:
      case clang::UO_Extension:
        os << ' ';
        break;
      case clang::UO_Plus:
      case clang::UO_Minus:
        if (clang::isa<clang::UnaryOperator>(Node->getSubExpr()))
          os << ' ';
        break;
      }
    }
    PrintExpr(Node->getSubExpr());

    if (Node->isPostfix())
      os << clang::UnaryOperator::getOpcodeStr(Node->getOpcode());
  }

  void VisitOffsetOfExpr(clang::OffsetOfExpr *Node) {
    os << "__builtin_offsetof(";
    Node->getTypeSourceInfo()->getType().print(os, Policy);
    os << ", ";
    bool PrintedSomething = false;
    for (unsigned i = 0, n = Node->getNumComponents(); i < n; ++i) {
      auto ON = Node->getComponent(i);
      if (ON.getKind() == clang::OffsetOfNode::Array) {
        // Array node
        os << '[';
        PrintExpr(Node->getIndexExpr(ON.getArrayExprIndex()));
        os << ']';
        PrintedSomething = true;
        continue;
      }

      // Skip implicit base indirections.
      if (ON.getKind() == clang::OffsetOfNode::Base)
        continue;

      // Field or identifier node.
      auto *Id = ON.getFieldName();
      if (!Id)
        continue;
      
      if (PrintedSomething)
        os << '.';
      else
        PrintedSomething = true;
      os << Id->getName();    
    }
    os << ')';
  }

  void VisitUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *Node){
    switch(Node->getKind()) {
    case clang::UETT_SizeOf:
      os << "sizeof";
      break;
    case clang::UETT_AlignOf:
      if (Policy.Alignof)
        os << "alignof";
      else if (Policy.UnderscoreAlignof)
        os << "_Alignof";
      else
        os << "__alignof";
      break;
    case clang::UETT_VecStep:
      os << "vec_step";
      break;
    default: llvm_unreachable("OpenMP is not supported.");
    }
    if (Node->isArgumentType()) {
      os << '(';
      Node->getArgumentType().print(os, Policy);
      os << ')';
    } else {
      os << ' ';
      PrintExpr(Node->getArgumentExpr());
    }
  }

  void VisitGenericSelectionExpr(clang::GenericSelectionExpr *Node) {
    os << "_Generic(";
    PrintExpr(Node->getControllingExpr());
    auto node = ultima::detail::generic_selection_expr_wrapper(Node);
    for (auto&& x : node) {
      os << ", ";
      auto T = x.get_type();
      if (T.isNull())
        os << "default";
      else
        T.print(os, Policy);
      os << ": ";
      PrintExpr(x.get_expr());
    }
    os << ')';
  }

  void VisitArraySubscriptExpr(clang::ArraySubscriptExpr *Node) {
    PrintExpr(Node->getLHS());
    os << '[';
    PrintExpr(Node->getRHS());
    os << ']';
  }

  static std::string to_identifier(const std::string& s){
    std::stringstream ss;
    for(auto c : s)switch(c){
    case '<': ss << "__left_angle__"; break;
    case '>': ss << "__right_angle__"; break;
    case '(': ss << "__left_paren__"; break;
    case ')': ss << "__right_paren__"; break;
    case ',': ss << "__comma__"; break;
    case '.': ss << "__dot__"; break;
    case ' ': break;
    case '*': ss << "__pointer__"; break;
    case '[': ss << "__left_square__"; break;
    case ']': ss << "__right_square__"; break;
    default: ss << c; break;
    }
    return ss.str();
  }

  void print_template_argument(const clang::TemplateArgument& targ){
    switch(targ.getKind()){
    case clang::TemplateArgument::ArgKind::Type:
      os << to_identifier(targ.getAsType()->getUnqualifiedDesugaredType()->getLocallyUnqualifiedSingleStepDesugaredType().getAsString(Policy));
      break;
    case clang::TemplateArgument::ArgKind::Expression:
      PrintExpr(targ.getAsExpr());
      break;
    case clang::TemplateArgument::ArgKind::Integral:
      os << targ.getAsIntegral();
      break;
    case clang::TemplateArgument::ArgKind::NullPtr:
      os << "NULL";
      break;
    case clang::TemplateArgument::ArgKind::Declaration:
      llvm_unreachable("Current ultima doesn't support declaration in template");
    case clang::TemplateArgument::ArgKind::Template:
      llvm_unreachable("Current ultima doesn't support template template parameter");
    case clang::TemplateArgument::ArgKind::Null:
      llvm_unreachable("Oops! something is wrong... /* clang::TemplateArgument::ArgKind::Null */");
    case clang::TemplateArgument::ArgKind::TemplateExpansion:
    case clang::TemplateArgument::ArgKind::Pack:
      llvm_unreachable("Current ultima doesn't support parameter pack");
    }
  }

  void print_template_arguments(const llvm::ArrayRef<clang::TemplateArgument>& arr){
    for(auto&& x : arr){
      os << to_identifier("<");
      print_template_argument(x);
      os << to_identifier(">");
    }
  }

  void print_template_arguments(const clang::TemplateArgumentList* tal){
    print_template_arguments(tal->asArray());
  }

  void PrintCallArgs(clang::CallExpr *Call) {
    for (unsigned i = 0, e = Call->getNumArgs(); i != e; ++i) {
      if (clang::isa<clang::CXXDefaultArgExpr>(Call->getArg(i))) {
        // Don't print any defaulted arguments
        break;
      }

      if (i) os << ", ";
      PrintExpr(Call->getArg(i));
    }
  }

  void VisitCallExpr(clang::CallExpr *Call) {
    if(auto f = clang::dyn_cast<clang::FunctionDecl>(Call->getCalleeDecl())){
      auto it = func_name.find(f);
      if(it != func_name.end())
        os << it->second;
      else{
        PrintExpr(Call->getCallee());
        if(auto list = f->getTemplateSpecializationArgs()){
          os << '_';
          print_template_arguments(list);
        }
      }
    }
    else
      PrintExpr(Call->getCallee());
    os << '(';
    PrintCallArgs(Call);
    os << ')';
  }
  void VisitMemberExpr(clang::MemberExpr *Node) {
    // FIXME: Suppress printing implicit bases (like "this")
    PrintExpr(Node->getBase());

    auto *ParentMember = clang::dyn_cast<clang::MemberExpr>(Node->getBase());
    auto *ParentDecl   = ParentMember
      ? clang::dyn_cast<clang::FieldDecl>(ParentMember->getMemberDecl()) : nullptr;

    if (!ParentDecl || !ParentDecl->isAnonymousStructOrUnion()){
      if(Node->isArrow())
        os << "->";
      else
        os << '.';
    }

    if (clang::FieldDecl *FD = clang::dyn_cast<clang::FieldDecl>(Node->getMemberDecl()))
      if (FD->isAnonymousStructOrUnion())
        return;

    if (auto *Qualifier = Node->getQualifier())
      Qualifier->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getMemberNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }
  void VisitExtVectorElementExpr(clang::ExtVectorElementExpr *Node) {
    PrintExpr(Node->getBase());
    os << '.';
    os << Node->getAccessor().getName();
  }
  const char* get_address_space_qualifier(clang::Expr* e){
    if(auto decl_ref_expr = clang::dyn_cast<clang::DeclRefExpr>(dig_expr(e)))
    if(auto decl = decl_ref_expr->getDecl()){
      if(has_annotation(decl, "cl_global"))
        return "__global ";
      else if(has_annotation(decl, "cl_local"))
        return "__local ";
    }
    return NULL;
  }
  void VisitCStyleCastExpr(clang::CStyleCastExpr *Node) {
    os << '(';
    if(auto asq = get_address_space_qualifier(Node->getSubExpr()))
      os << asq;
    Node->getTypeAsWritten().print(os, Policy);
    os << ')';
    PrintExpr(Node->getSubExpr());
  }
  void VisitCompoundLiteralExpr(clang::CompoundLiteralExpr *Node) {
    os << '(';
    Node->getType().print(os, Policy);
    os << ')';
    PrintExpr(Node->getInitializer());
  }
  void VisitImplicitCastExpr(clang::ImplicitCastExpr *Node) {
    PrintExpr(Node->getSubExpr());
  }
  void VisitBinaryOperator(clang::BinaryOperator *Node) {
    PrintExpr(Node->getLHS());
    os << ' ' << clang::BinaryOperator::getOpcodeStr(Node->getOpcode()) << ' ';
    PrintExpr(Node->getRHS());
  }
  void VisitCompoundAssignOperator(clang::CompoundAssignOperator *Node) {
    PrintExpr(Node->getLHS());
    os << ' ' << clang::BinaryOperator::getOpcodeStr(Node->getOpcode()) << ' ';
    PrintExpr(Node->getRHS());
  }
  void VisitConditionalOperator(clang::ConditionalOperator *Node) {
    PrintExpr(Node->getCond());
    os << " ? ";
    PrintExpr(Node->getLHS());
    os << " : ";
    PrintExpr(Node->getRHS());
  }

  // GNU extensions.

  void VisitBinaryConditionalOperator(clang::BinaryConditionalOperator *Node) {
    PrintExpr(Node->getCommon());
    os << " ?: ";
    PrintExpr(Node->getFalseExpr());
  }
  void VisitAddrLabelExpr(clang::AddrLabelExpr *Node) {
    os << "&&" << Node->getLabel()->getName();
  }

  void VisitStmtExpr(clang::StmtExpr *E) {
    os << '(';
    PrintRawCompoundStmt(E->getSubStmt());
    os << ')';
  }

  void VisitChooseExpr(clang::ChooseExpr *Node) {
    os << "__builtin_choose_expr(";
    PrintExpr(Node->getCond());
    os << ", ";
    PrintExpr(Node->getLHS());
    os << ", ";
    PrintExpr(Node->getRHS());
    os << ')';
  }

  void VisitGNUNullExpr(clang::GNUNullExpr *) {
    os << "__null";
  }

  void VisitShuffleVectorExpr(clang::ShuffleVectorExpr *Node) {
    os << "__builtin_shufflevector(";
    for (unsigned i = 0, e = Node->getNumSubExprs(); i != e; ++i) {
      if (i) os << ", ";
      PrintExpr(Node->getExpr(i));
    }
    os << ')';
  }

  void VisitConvertVectorExpr(clang::ConvertVectorExpr *Node) {
    os << "__builtin_convertvector(";
    PrintExpr(Node->getSrcExpr());
    os << ", ";
    Node->getType().print(os, Policy);
    os << ')';
  }

  void VisitInitListExpr(clang::InitListExpr* Node) {
    if (Node->getSyntacticForm()) {
      Visit(Node->getSyntacticForm());
      return;
    }

    os << '{';
    for (unsigned i = 0, e = Node->getNumInits(); i != e; ++i) {
      if (i) os << ", ";
      if (Node->getInit(i))
        PrintExpr(Node->getInit(i));
      else
        os << "{}";
    }
    os << '}';
  }

  void VisitArrayInitLoopExpr(clang::ArrayInitLoopExpr *Node) {
    // There's no way to express this expression in any of our supported
    // languages, so just emit something terse and (hopefully) clear.
    os << '{';
    PrintExpr(Node->getSubExpr());
    os << '}';
  }

  void VisitArrayInitIndexExpr(clang::ArrayInitIndexExpr*) {
    os << '*';
  }

  void VisitParenListExpr(clang::ParenListExpr* Node) {
    os << '(';
    for (unsigned i = 0, e = Node->getNumExprs(); i != e; ++i) {
      if (i) os << ", ";
      PrintExpr(Node->getExpr(i));
    }
    os << ')';
  }

  void VisitDesignatedInitExpr(clang::DesignatedInitExpr *Node) {
    bool NeedsEquals = true;
    for (const auto &D : Node->designators()) {
      if (D.isFieldDesignator()) {
        if (D.getDotLoc().isInvalid()) {
          if (auto *II = D.getFieldName()) {
            os << II->getName() << ':';
            NeedsEquals = false;
          }
        } else {
          os << '.' << D.getFieldName()->getName();
        }
      } else {
        os << '[';
        if (D.isArrayDesignator()) {
          PrintExpr(Node->getArrayIndex(D));
        } else {
          PrintExpr(Node->getArrayRangeStart(D));
          os << " ... ";
          PrintExpr(Node->getArrayRangeEnd(D));
        }
        os << ']';
      }
    }

    if (NeedsEquals)
      os << " = ";
    else
      os << ' ';
    PrintExpr(Node->getInit());
  }

  void VisitDesignatedInitUpdateExpr(
      clang::DesignatedInitUpdateExpr *Node) {
    os << '{';
    os << "/*base*/";
    PrintExpr(Node->getBase());
    os << ", ";

    os << "/*updater*/";
    PrintExpr(Node->getUpdater());
    os << '}';
  }

  void VisitNoInitExpr(clang::NoInitExpr*) {}

  void VisitImplicitValueInitExpr(clang::ImplicitValueInitExpr *Node) {
    if (Node->getType()->getAsCXXRecordDecl()) {
      os << "/*implicit*/";
      Node->getType().print(os, Policy);
      os << "()";
    } else {
      os << "/*implicit*/(";
      Node->getType().print(os, Policy);
      os << ')';
      if (Node->getType()->isRecordType())
        os << "{}";
      else
        os << 0;
    }
  }

  void VisitVAArgExpr(clang::VAArgExpr *Node) {
    os << "__builtin_va_arg(";
    PrintExpr(Node->getSubExpr());
    os << ", ";
    Node->getType().print(os, Policy);
    os << ')';
  }

  void VisitPseudoObjectExpr(clang::PseudoObjectExpr *Node) {
    PrintExpr(Node->getSyntacticForm());
  }

  void VisitAtomicExpr(clang::AtomicExpr *Node) {
    const char *Name = nullptr;
    switch (Node->getOp()) {
      using namespace clang;
#define BUILTIN(ID, TYPE, ATTRS)
#define ATOMIC_BUILTIN(ID, TYPE, ATTRS) \
    case AtomicExpr::AO ## ID: \
      Name = #ID "("; \
      break;
#include "clang/Basic/Builtins.def"
#undef ATOMIC_BUILTIN
#undef BUILTIN
    }
    os << Name;

    // AtomicExpr stores its subexpressions in a permuted order.
    PrintExpr(Node->getPtr());
    if (Node->getOp() != clang::AtomicExpr::AO__c11_atomic_load &&
        Node->getOp() != clang::AtomicExpr::AO__atomic_load_n) {
      os << ", ";
      PrintExpr(Node->getVal1());
    }
    if (Node->getOp() == clang::AtomicExpr::AO__atomic_exchange ||
        Node->isCmpXChg()) {
      os << ", ";
      PrintExpr(Node->getVal2());
    }
    if (Node->getOp() == clang::AtomicExpr::AO__atomic_compare_exchange ||
        Node->getOp() == clang::AtomicExpr::AO__atomic_compare_exchange_n) {
      os << ", ";
      PrintExpr(Node->getWeak());
    }
    if (Node->getOp() != clang::AtomicExpr::AO__c11_atomic_init) {
      os << ", ";
      PrintExpr(Node->getOrder());
    }
    if (Node->isCmpXChg()) {
      os << ", ";
      PrintExpr(Node->getOrderFail());
    }
    os << ')';
  }

  // C++
  void VisitCXXOperatorCallExpr(clang::CXXOperatorCallExpr *Node) {
    const char *OpStrings[clang::NUM_OVERLOADED_OPERATORS] = {
      "",
#define OVERLOADED_OPERATOR(Name,Spelling,Token,Unary,Binary,MemberOnly) \
      Spelling,
#include "clang/Basic/OperatorKinds.def"
#undef OVERLOADED_OPERATOR
    };

    auto Kind = Node->getOperator();
    if (Kind == clang::OO_PlusPlus || Kind == clang::OO_MinusMinus) {
      if (Node->getNumArgs() == 1) {
        os << OpStrings[Kind] << ' ';
        PrintExpr(Node->getArg(0));
      } else {
        PrintExpr(Node->getArg(0));
        os << ' ' << OpStrings[Kind];
      }
    } else if (Kind == clang::OO_Arrow) {
      PrintExpr(Node->getArg(0));
    } else if (Kind == clang::OO_Call) {
      PrintExpr(Node->getArg(0));
      os << '(';
      for (unsigned ArgIdx = 1; ArgIdx < Node->getNumArgs(); ++ArgIdx) {
        if (ArgIdx > 1)
          os << ", ";
        if (!clang::isa<clang::CXXDefaultArgExpr>(Node->getArg(ArgIdx)))
          PrintExpr(Node->getArg(ArgIdx));
      }
      os << ')';
    } else if (Kind == clang::OO_Subscript) {
      if(auto CArray = Node->getArg(0)->getType()->getAs<clang::TemplateSpecializationType>())
      if(CArray->getTemplateName().getAsTemplateDecl()->getQualifiedNameAsString() == "CArray"){
        const auto decl_ref = clang::dyn_cast<clang::DeclRefExpr>(dig_expr(Node->getArg(0)));
        if(decl_ref == nullptr)
          throw std::runtime_error("Current ultima only support array subscription with 'raw' CArray object.");
        const auto name = decl_ref->getNameInfo().getAsString();
        auto var_info = std::find_if(func_arg_info.back().begin(), func_arg_info.back().end(), [name](const function_special_argument_info& t){
          return t.name == name && t.arg_flag == function_special_argument_info::raw;
        });
        if(var_info == func_arg_info.back().end())
          throw std::runtime_error("only 'raw' CArray can subscript");
        os << name << "[get_CArrayIndex";
        if(var_info->ndim > 1){
          const auto type = Node->getArg(1)->getType();
          if(auto array = type->getAsArrayTypeUnsafe())
            os << "Raw_" << var_info->ndim << '_' << to_identifier('<'+array->getElementType()->getUnqualifiedDesugaredType()->getLocallyUnqualifiedSingleStepDesugaredType().getAsString()+'>');
          else if(auto pointer = type->getAs<clang::PointerType>())
            os << "Raw_" << var_info->ndim << '_' << to_identifier('<'+pointer->getPointeeType()->getUnqualifiedDesugaredType()->getLocallyUnqualifiedSingleStepDesugaredType().getAsString()+'>');
          else
            os << "I_" << var_info->ndim;
        }
        else
          os << "Raw_" << var_info->ndim;
        os << "(&" << name << "_info, ";
        PrintExpr(Node->getArg(1));
        os << ")/sizeof(" << var_info->type << ")]";
        return;
      }
      PrintExpr(Node->getArg(0));
      os << '[';
      PrintExpr(Node->getArg(1));
      os << ']';
    } else if (Node->getNumArgs() == 1) {
      os << OpStrings[Kind] << ' ';
      PrintExpr(Node->getArg(0));
    } else if (Node->getNumArgs() == 2) {
      PrintExpr(Node->getArg(0));
      os << ' ' << OpStrings[Kind] << ' ';
      PrintExpr(Node->getArg(1));
    } else {
      llvm_unreachable("unknown overloaded operator");
    }
  }

  static clang::Expr* dig_expr(clang::Expr* e){
    if(auto p = clang::dyn_cast<clang::ParenExpr>(e))
      return dig_expr(p->getSubExpr());
    if(auto ic = clang::dyn_cast<clang::ImplicitCastExpr>(e))
      return dig_expr(ic->getSubExpr());
    return e;
  }

  void VisitCXXMemberCallExpr(clang::CXXMemberCallExpr *Node) {
    auto member_expr = clang::dyn_cast<clang::MemberExpr>(Node->getCallee());
    auto base = member_expr->getBase();
    auto base_type = base->getType()->getUnqualifiedDesugaredType()->getAs<clang::RecordType>();
    if(base_type){
      if(base_type->getDecl()->getName() == "CIndexer"
      && member_expr->getMemberNameInfo().getAsString() == "size"){
        auto ind = dig_expr(base);
        if(!clang::isa<clang::DeclRefExpr>(ind))
          throw std::runtime_error("Current ultima only support calling CIndexer::size() with a CIndexer object.");
        Visit(ind);
        os << "_size";
        return;
      }
      else if(base_type->getDecl()->getName() == "CArray"){
        auto raw = clang::dyn_cast<clang::DeclRefExpr>(dig_expr(base));
        const auto member_function_name = member_expr->getMemberNameInfo().getAsString();
        if(member_function_name == "size"){
          if(raw == nullptr)
            throw std::runtime_error("Current ultima only support calling CArray::size() with a CArray object.");
          os << "((const size_t)";
          Visit(raw);
          os << "_info.size_)";
          return;
        }
        else if(member_function_name == "shape" || member_function_name == "strides"){
          if(raw == nullptr)
            throw std::runtime_error("Current ultima only support calling CArray::" + member_function_name + "() with a CArray object.");
          const auto name = raw->getNameInfo().getAsString();
          auto var_info = std::find_if(func_arg_info.back().begin(), func_arg_info.back().end(), [name](const function_special_argument_info& t){
            return t.name == name && t.arg_flag == function_special_argument_info::raw;
          });
          if(var_info == func_arg_info.back().end())
            throw std::runtime_error("only 'raw' CArray can be used to call member function");
          os << "((const size_t*)";
          if(var_info->ndim == 0)
            os << "NULL)";
          else{
            if(var_info->ndim == 1)
              os << '&';
            Visit(raw);
            os << "_info." << member_function_name << "_)";
          }
          return;
        }
      }
    }
    throw std::runtime_error("current ultima doesn't support member function call.");
  }

  void VisitCXXNamedCastExpr(clang::CXXNamedCastExpr *Node) {
    os << '(';
    if(auto asq = get_address_space_qualifier(Node->getSubExpr()))
      os << asq;
    Node->getTypeAsWritten().print(os, Policy);
    os << ")(";
    PrintExpr(Node->getSubExpr());
    os << ')';
  }

  void VisitCXXTypeidExpr(clang::CXXTypeidExpr *Node) {
    os << "typeid(";
    if (Node->isTypeOperand()) {
      Node->getTypeOperandSourceInfo()->getType().print(os, Policy);
    } else {
      PrintExpr(Node->getExprOperand());
    }
    os << ')';
  }

  void VisitCXXUuidofExpr(clang::CXXUuidofExpr *Node) {
    os << "__uuidof(";
    if (Node->isTypeOperand()) {
      Node->getTypeOperandSourceInfo()->getType().print(os, Policy);
    } else {
      PrintExpr(Node->getExprOperand());
    }
    os << ')';
  }

  void VisitMSPropertyRefExpr(clang::MSPropertyRefExpr *Node) {
    PrintExpr(Node->getBaseExpr());
    if (Node->isArrow())
      os << "->";
    else
      os << '.';
    if (auto *Qualifier =
        Node->getQualifierLoc().getNestedNameSpecifier())
      Qualifier->print(os, Policy);
    os << Node->getPropertyDecl()->getDeclName();
  }

  void VisitMSPropertySubscriptExpr(clang::MSPropertySubscriptExpr *Node) {
    PrintExpr(Node->getBase());
    os << '[';
    PrintExpr(Node->getIdx());
    os << ']';
  }

  void VisitUserDefinedLiteral(clang::UserDefinedLiteral *Node) {
    switch (Node->getLiteralOperatorKind()) {
    case clang::UserDefinedLiteral::LOK_Raw:
      os << clang::cast<clang::StringLiteral>(Node->getArg(0)->IgnoreImpCasts())->getString();
      break;
    case clang::UserDefinedLiteral::LOK_Template: {
      auto *DRE = clang::cast<clang::DeclRefExpr>(Node->getCallee()->IgnoreImpCasts());
      const auto *Args =
        clang::cast<clang::FunctionDecl>(DRE->getDecl())->getTemplateSpecializationArgs();
      assert(Args);

      if (Args->size() != 1) {
        os << "operator\"\"" << Node->getUDSuffix()->getName();
        clang::printTemplateArgumentList(os, Args->asArray(), Policy);
        os << "()";
        return;
      }

      const auto &Pack = Args->get(0);
      for (const auto &P : Pack.pack_elements()) {
        char C = (char)P.getAsIntegral().getZExtValue();
        os << C;
      }
      break;
    }
    case clang::UserDefinedLiteral::LOK_Integer: {
      // Print integer literal without suffix.
      auto *Int = clang::cast<clang::IntegerLiteral>(Node->getCookedLiteral());
      os << Int->getValue().toString(10, /*isSigned*/false);
      break;
    }
    case clang::UserDefinedLiteral::LOK_Floating: {
      // Print floating literal without suffix.
      auto *Float = clang::cast<clang::FloatingLiteral>(Node->getCookedLiteral());
      PrintFloatingLiteral(os, Float, /*PrintSuffix=*/false);
      break;
    }
    case clang::UserDefinedLiteral::LOK_String:
    case clang::UserDefinedLiteral::LOK_Character:
      PrintExpr(Node->getCookedLiteral());
      break;
    }
    os << Node->getUDSuffix()->getName();
  }

  void VisitCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *Node) {
    os << (Node->getValue() ? "true" : "false");
  }

  void VisitCXXNullPtrLiteralExpr(clang::CXXNullPtrLiteralExpr*) {
    os << "nullptr";
  }

  void VisitCXXThisExpr(clang::CXXThisExpr*) {
    os << "this";
  }

  void VisitCXXThrowExpr(clang::CXXThrowExpr *Node) {
    if (!Node->getSubExpr())
      os << "throw";
    else {
      os << "throw ";
      PrintExpr(Node->getSubExpr());
    }
  }

  void VisitCXXDefaultArgExpr(clang::CXXDefaultArgExpr*) {
    // Nothing to print: we picked up the default argument.
  }

  void VisitCXXDefaultInitExpr(clang::CXXDefaultInitExpr*) {
    // Nothing to print: we picked up the default initializer.
  }

  void VisitCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *Node) {
    os << '(';
    if(auto asq = get_address_space_qualifier(Node->getSubExpr()))
      os << asq;
    Node->getTypeAsWritten().print(os, Policy);
    os << ')';
    // If there are no parens, this is list-initialization, and the braces are
    // part of the syntax of the inner construct.
    if (Node->getLParenLoc().isValid())
      os << '(';
    PrintExpr(Node->getSubExpr());
    if (Node->getLParenLoc().isValid())
      os << ')';
  }

  void VisitCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *Node) {
    PrintExpr(Node->getSubExpr());
  }

  void VisitCXXTemporaryObjectExpr(clang::CXXTemporaryObjectExpr *Node) {
    Node->getType().print(os, Policy);
    if (Node->isStdInitListInitialization())
      /* Nothing to do; braces are part of creating the std::initializer_list. */;
    else if (Node->isListInitialization())
      os << '{';
    else
      os << '(';
    for (auto Arg = Node->arg_begin(), ArgEnd = Node->arg_end();
         Arg != ArgEnd; ++Arg) {
      if ((*Arg)->isDefaultArgument())
        break;
      if (Arg != Node->arg_begin())
        os << ", ";
      PrintExpr(*Arg);
    }
    if (Node->isStdInitListInitialization())
      /* See above. */;
    else if (Node->isListInitialization())
      os << '}';
    else
      os << ')';
  }

  void VisitLambdaExpr(clang::LambdaExpr *Node) {
    os << '[';
    bool NeedComma = false;
    switch (Node->getCaptureDefault()) {
    case clang::LCD_None:
      break;

    case clang::LCD_ByCopy:
      os << '=';
      NeedComma = true;
      break;

    case clang::LCD_ByRef:
      os << '&';
      NeedComma = true;
      break;
    }
    for (auto&& C : Node->explicit_captures()) {
      if (NeedComma)
        os << ", ";
      NeedComma = true;

      switch (C.getCaptureKind()) {
      case clang::LCK_This:
        os << "this";
        break;
      case clang::LCK_StarThis:
        os << "*this";
        break;
      case clang::LCK_ByRef:
        if (Node->getCaptureDefault() != clang::LCD_ByRef || Node->isInitCapture(&C))
          os << '&';
        os << C.getCapturedVar()->getName();
        break;

      case clang::LCK_ByCopy:
        os << C.getCapturedVar()->getName();
        break;
      case clang::LCK_VLAType:
        llvm_unreachable("VLA type in explicit captures.");
      }

      if (Node->isInitCapture(&C))
        PrintExpr(C.getCapturedVar()->getInit());
    }
    os << ']';

    if (Node->hasExplicitParameters()) {
      os << " (";
      auto *Method = Node->getCallOperator();
      NeedComma = false;
      for (auto P : Method->parameters()) {
        if (NeedComma) {
          os << ", ";
        } else {
          NeedComma = true;
        }
        std::string ParamStr = P->getNameAsString();
        P->getOriginalType().print(os, Policy, ParamStr);
      }
      if (Method->isVariadic()) {
        if (NeedComma)
          os << ", ";
        os << "...";
      }
      os << ')';

      if (Node->isMutable())
        os << " mutable";

      const auto *Proto
        = Method->getType()->getAs<clang::FunctionProtoType>();
      Proto->printExceptionSpecification(os, Policy);

      // FIXME: Attributes

      // Print the trailing return type if it was specified in the source.
      if (Node->hasExplicitResultType()) {
        os << " -> ";
        Proto->getReturnType().print(os, Policy);
      }
    }

    // Print the body.
    auto *Body = Node->getBody();
    os << ' ';
    PrintStmt(Body);
  }

  void VisitCXXScalarValueInitExpr(clang::CXXScalarValueInitExpr *Node) {
    if (auto *TSInfo = Node->getTypeSourceInfo())
      TSInfo->getType().print(os, Policy);
    else
      Node->getType().print(os, Policy);
    os << "()";
  }

  void VisitCXXNewExpr(clang::CXXNewExpr *E) {
    if (E->isGlobalNew())
      os << "::";
    os << "new ";
    unsigned NumPlace = E->getNumPlacementArgs();
    if (NumPlace > 0 && !clang::isa<clang::CXXDefaultArgExpr>(E->getPlacementArg(0))) {
      os << '(';
      PrintExpr(E->getPlacementArg(0));
      for (unsigned i = 1; i < NumPlace; ++i) {
        if (clang::isa<clang::CXXDefaultArgExpr>(E->getPlacementArg(i)))
          break;
        os << ", ";
        PrintExpr(E->getPlacementArg(i));
      }
      os << ") ";
    }
    if (E->isParenTypeId())
      os << '(';
    std::string TypeS;
    if (clang::Expr *Size = ultima::detail::get_array_size(E)) {
      llvm::raw_string_ostream s(TypeS);
      s << '[';
      Visit(Size);
      s << ']';
    }
    E->getAllocatedType().print(os, Policy, TypeS);
    if (E->isParenTypeId())
      os << ')';

    auto InitStyle = E->getInitializationStyle();
    if (InitStyle) {
      if (InitStyle == clang::CXXNewExpr::CallInit)
        os << '(';
      PrintExpr(E->getInitializer());
      if (InitStyle == clang::CXXNewExpr::CallInit)
        os << ')';
    }
  }

  void VisitCXXDeleteExpr(clang::CXXDeleteExpr *E) {
    if (E->isGlobalDelete())
      os << "::";
    os << "delete ";
    if (E->isArrayForm())
      os << "[] ";
    PrintExpr(E->getArgument());
  }

  void VisitCXXPseudoDestructorExpr(clang::CXXPseudoDestructorExpr *E) {
    PrintExpr(E->getBase());
    if (E->isArrow())
      os << "->";
    else
      os << '.';
    if (E->getQualifier())
      E->getQualifier()->print(os, Policy);
    os << '~';

    if (auto *II = E->getDestroyedTypeIdentifier())
      os << II->getName();
    else
      E->getDestroyedType().print(os, Policy);
  }

  void VisitCXXConstructExpr(clang::CXXConstructExpr *E, const char* name = nullptr) {
    if(E->isElidable()){
      auto a = E->getArg(0);
      if(auto subexpr = clang::dyn_cast<clang::CXXConstructExpr>(a->IgnoreImplicit()))
        VisitCXXConstructExpr(subexpr, name);
      else
        Visit(a);
      return;
    }

    const bool is_defaulted = E->getConstructor()->isDefaulted();

    if (is_defaulted && E->isListInitialization() && !E->isStdInitListInitialization())
      os << '{';

    auto f = clang::dyn_cast<clang::FunctionDecl>(E->getConstructor());

    if(!is_defaulted){
      if(f){
        auto it = func_name.find(f);
        if(it != func_name.end())
          os << it->second;
        else{
          os << "constructor_" << to_identifier(E->getConstructor()->getDeclName().getAsString());
          if(auto list = f->getTemplateSpecializationArgs()){
            os << '_';
            print_template_arguments(list);
          }
        }
        os << '(';
      }

      if(name)
        os << '&' << name;
    }

    for (unsigned i = 0, e = E->getNumArgs(); i != e; ++i) {
      if (clang::isa<clang::CXXDefaultArgExpr>(E->getArg(i))) {
        // Don't print any defaulted arguments
        break;
      }

      if ((!is_defaulted && name) || i) os << ", ";
      PrintExpr(E->getArg(i));
    }

    if(!is_defaulted){
      if(f)
        os << ')';
    }
    else if (E->isListInitialization() && !E->isStdInitListInitialization())
      os << '}';
  }

  void VisitCXXInheritedCtorInitExpr(clang::CXXInheritedCtorInitExpr*) {
    // Parens are printed by the surrounding context.
    os << "<forwarded>";
  }

  void VisitCXXStdInitializerListExpr(clang::CXXStdInitializerListExpr *E) {
    PrintExpr(E->getSubExpr());
  }

  void VisitExprWithCleanups(clang::ExprWithCleanups *E) {
    // Just forward to the subexpression.
    PrintExpr(E->getSubExpr());
  }

  void VisitCXXUnresolvedConstructExpr(clang::CXXUnresolvedConstructExpr *Node) {
    Node->getTypeAsWritten().print(os, Policy);
    os << '(';
    for (auto Arg = Node->arg_begin(), ArgEnd = Node->arg_end();
         Arg != ArgEnd; ++Arg) {
      if (Arg != Node->arg_begin())
        os << ", ";
      PrintExpr(*Arg);
    }
    os << ')';
  }

  void VisitCXXDependentScopeMemberExpr(clang::CXXDependentScopeMemberExpr *Node) {
    if (!Node->isImplicitAccess()) {
      PrintExpr(Node->getBase());
      if(Node->isArrow())
        os << "->";
      else
        os << '.';
    }
    if (auto *Qualifier = Node->getQualifier())
      Qualifier->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getMemberNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }

  void VisitUnresolvedMemberExpr(clang::UnresolvedMemberExpr *Node) {
    if (!Node->isImplicitAccess()) {
      PrintExpr(Node->getBase());
      if(Node->isArrow())
        os << "->";
      else
        os << '.';
    }
    if (auto *Qualifier = Node->getQualifier())
      Qualifier->print(os, Policy);
    if (Node->hasTemplateKeyword())
      os << "template ";
    os << Node->getMemberNameInfo();
    if (Node->hasExplicitTemplateArgs())
      clang::printTemplateArgumentList(os, Node->template_arguments(), Policy);
  }

  static const char *getTypeTraitName(clang::TypeTrait TT) {
    switch (TT) {
#define TYPE_TRAIT_1(Spelling, Name, Key) \
  case clang::UTT_##Name: return #Spelling;
#define TYPE_TRAIT_2(Spelling, Name, Key) \
  case clang::BTT_##Name: return #Spelling;
#define TYPE_TRAIT_N(Spelling, Name, Key) \
    case clang::TT_##Name: return #Spelling;
#include "clang/Basic/TokenKinds.def"
#undef TYPE_TRAIT_N
#undef TYPE_TRAIT_2
#undef TYPE_TRAIT_1
    }
    llvm_unreachable("Type trait not covered by switch");
  }

  static const char *getTypeTraitName(clang::ArrayTypeTrait ATT) {
    switch (ATT) {
    case clang::ATT_ArrayRank:        return "__array_rank";
    case clang::ATT_ArrayExtent:      return "__array_extent";
    }
    llvm_unreachable("Array type trait not covered by switch");
  }

  static const char *getExpressionTraitName(clang::ExpressionTrait ET) {
    switch (ET) {
    case clang::ET_IsLValueExpr:      return "__is_lvalue_expr";
    case clang::ET_IsRValueExpr:      return "__is_rvalue_expr";
    }
    llvm_unreachable("Expression type trait not covered by switch");
  }

  void VisitTypeTraitExpr(clang::TypeTraitExpr *E) {
    os << getTypeTraitName(E->getTrait()) << '(';
    for (unsigned I = 0, N = E->getNumArgs(); I != N; ++I) {
      if (I > 0)
        os << ", ";
      E->getArg(I)->getType().print(os, Policy);
    }
    os << ')';
  }

  void VisitArrayTypeTraitExpr(clang::ArrayTypeTraitExpr *E) {
    os << getTypeTraitName(E->getTrait()) << '(';
    E->getQueriedType().print(os, Policy);
    os << ')';
  }

  void VisitExpressionTraitExpr(clang::ExpressionTraitExpr *E) {
    os << getExpressionTraitName(E->getTrait()) << '(';
    PrintExpr(E->getQueriedExpression());
    os << ')';
  }

  void VisitCXXNoexceptExpr(clang::CXXNoexceptExpr *E) {
    os << "noexcept(";
    PrintExpr(E->getOperand());
    os << ')';
  }

  void VisitPackExpansionExpr(clang::PackExpansionExpr *E) {
    PrintExpr(E->getPattern());
    os << "...";
  }

  void VisitSizeOfPackExpr(clang::SizeOfPackExpr *E) {
    os << "sizeof...(" << *E->getPack() << ')';
  }

  void VisitSubstNonTypeTemplateParmPackExpr(clang::SubstNonTypeTemplateParmPackExpr *Node) {
    os << *Node->getParameterPack();
  }

  void VisitSubstNonTypeTemplateParmExpr(clang::SubstNonTypeTemplateParmExpr *Node) {
    Visit(Node->getReplacement());
  }

  void VisitFunctionParmPackExpr(clang::FunctionParmPackExpr *E) {
    os << *E->getParameterPack();
  }

  void VisitMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *Node){
    PrintExpr(Node->GetTemporaryExpr());
  }

  void VisitCXXFoldExpr(clang::CXXFoldExpr *E) {
    os << '(';
    if (E->getLHS()) {
      PrintExpr(E->getLHS());
      os << ' ' << clang::BinaryOperator::getOpcodeStr(E->getOperator()) << ' ';
    }
    os << "...";
    if (E->getRHS()) {
      os << ' ' << clang::BinaryOperator::getOpcodeStr(E->getOperator()) << ' ';
      PrintExpr(E->getRHS());
    }
    os << ')';
  }

  // C++ Coroutines TS

  void VisitCoroutineBodyStmt(clang::CoroutineBodyStmt *S) {
    Visit(S->getBody());
  }

  void VisitCoreturnStmt(clang::CoreturnStmt *S) {
    os << "co_return";
    if (S->getOperand()) {
      os << ' ';
      Visit(S->getOperand());
    }
    os << ';';
  }

  void VisitCoawaitExpr(clang::CoawaitExpr *S) {
    os << "co_await ";
    PrintExpr(S->getOperand());
  }


/*  void VisitDependentCoawaitExpr(clang::DependentCoawaitExpr *S) {
    os << "co_await ";
    PrintExpr(S->getOperand());
  }*/


  void VisitCoyieldExpr(clang::CoyieldExpr *S) {
    os << "co_yield ";
    PrintExpr(S->getOperand());
  }

  void VisitBlockExpr(clang::BlockExpr *Node) {
    auto *BD = Node->getBlockDecl();
    os << '^';

    const auto *AFT = Node->getFunctionType();

    if (clang::isa<clang::FunctionNoProtoType>(AFT)) {
      os << "()";
    } else if (!BD->param_empty() || clang::cast<clang::FunctionProtoType>(AFT)->isVariadic()) {
      os << '(';
      for (auto AI = BD->param_begin(),
           E = BD->param_end(); AI != E; ++AI) {
        if (AI != BD->param_begin()) os << ", ";
        std::string ParamStr = (*AI)->getNameAsString();
        (*AI)->getType().print(os, Policy, ParamStr);
      }

      const auto *FT = clang::cast<clang::FunctionProtoType>(AFT);
      if (FT->isVariadic()) {
        if (!BD->param_empty()) os << ", ";
        os << "...";
      }
      os << ')';
    }
    os << "{ }";
  }

  void VisitOpaqueValueExpr(clang::OpaqueValueExpr *Node) { 
    PrintExpr(Node->getSourceExpr());
  }

  void VisitTypoExpr(clang::TypoExpr*) {
    // TODO: Print something reasonable for a TypoExpr, if necessary.
    llvm_unreachable("Cannot print TypoExpr nodes");
  }

  void VisitAsTypeExpr(clang::AsTypeExpr *Node) {
    os << "__builtin_astype(";
    PrintExpr(Node->getSrcExpr());
    os << ", ";
    Node->getType().print(os, Policy);
    os << ')';
  }
};

class decl_visitor : public clang::DeclVisitor<decl_visitor>{
  llvm::raw_ostream& indent() { return indent(indentation); }
  static bool has_annotation(clang::Decl* decl, llvm::StringRef s){
    return stmt_visitor::has_annotation(decl, s);
  }
  template<typename T>
  static clang::CXXRecordDecl* get_unnamed_record_decl(T* f){
    const clang::Type* t = f->getType().split().Ty;
    if(t->getTypeClass() == clang::Type::Elaborated){
      auto r = clang::cast<clang::ElaboratedType>(t)->getNamedType()->getAsCXXRecordDecl();
      if(r && !r->getIdentifier())
        return r;
    }
    return nullptr;
  }
  template<typename T>
  static clang::EnumDecl* get_unnamed_enum_decl(T* f){
    const clang::Type* t = f->getType().split().Ty;
    if(t->getTypeClass() == clang::Type::Elaborated){
      auto r_ = clang::cast<clang::ElaboratedType>(t)->getNamedType()->getAsTagDecl();
      auto r = clang::dyn_cast<clang::EnumDecl>(r_);
      if(r && !r->getIdentifier())
        return r;
    }
    return nullptr;
  }

  ostreams os;
  clang::PrintingPolicy policy;
  unsigned indentation;
  std::vector<std::vector<function_special_argument_info>> func_arg_info;
  std::unordered_map<clang::FunctionDecl*, std::string> func_name;
  stmt_visitor sv;
  int print_out_counter = 0;
  std::vector<std::string> delayed_outputs;

public:
  decl_visitor(llvm::raw_ostream& os, const clang::PrintingPolicy& policy,
               unsigned indentation = 0)
    : os(os), policy(policy), indentation(indentation),
      sv{this->os, this->policy, this->indentation, *this, this->func_arg_info, this->func_name, this->delayed_outputs} { }

  llvm::raw_ostream& indent(unsigned indentation) {
    for (unsigned i = 0; i != indentation; ++i)
      os << "  ";
    return os;
  }

  static const clang::AttrVec& get_attrs(const clang::Decl* d){
    if(d->hasAttrs())
      return d->getAttrs();
    else{
      static clang::AttrVec dummy(0);
      return dummy;
    }
  }

  static llvm::ArrayRef<const clang::Attr*> get_attrs(const clang::AttributedStmt* s){
    return s->getAttrs();
  }

  template<typename T>
  void prettyPrintAttributes(T* D){
    if (policy.PolishForDeclaration)
      return;

    for (auto *x : get_attrs(D)) {
      if (x->isInherited() || x->isImplicit())
        continue;
      if(auto aa = clang::dyn_cast<clang::AnnotateAttr>(x))
        continue;
      switch (x->getKind()) {
#define ATTR(X)
#define PRAGMA_SPELLING_ATTR(X) case clang::attr::X:
#include "clang/Basic/AttrList.inc"
#undef PRAGMA_SPELLING_ATTR
#undef ATTR
        break;
      default:
        x->printPretty(os, policy);
        break;
      }
    }
  }
  template<typename T>
  std::vector<std::string> prettyPrintPragmas(T* D){
    std::vector<std::string> annons;
    if (policy.PolishForDeclaration)
      return annons;

    for (auto* x : get_attrs(D)) {
      if(auto aa = clang::dyn_cast<clang::AnnotateAttr>(x)){
        auto an = aa->getAnnotation();
        if(an == "cl_global")
          os << "__global ";
        else if(an == "cl_kernel")
          os << "__kernel ";
        else if(an == "cl_local")
          os << "__local ";
        else if(an == "cu_global");
        else if(an == "cu_device");
        else if(an == "cu_shared");
        else
          annons.emplace_back(an);
        continue;
      }
      switch (x->getKind()) {
#define ATTR(X)
#define PRAGMA_SPELLING_ATTR(X) case clang::attr::X:
#include "clang/Basic/AttrList.inc"
#undef PRAGMA_SPELLING_ATTR
#undef ATTR
        x->printPretty(os, policy);
        indent();
        break;
      default:
        break;
      }
    }
    return annons;
  }

  void printGroup(clang::Decl** Begin, unsigned NumDecls) {
    delayed_outputs.push_back("");

    if (NumDecls == 1) {
      Visit(*Begin);
      return;
    }
    clang::Decl** End = Begin + NumDecls;
    auto backup = policy;
    auto TD = clang::isa<clang::TagDecl>(*Begin);
    if (TD)
      ++Begin;

    bool isFirst = true;
    for ( ; Begin != End; ++Begin) {
      if (isFirst){
        if(TD)
          policy.IncludeTagDefinition = true;
        policy.SuppressSpecifiers = false;
        isFirst = false;
      }
      else{
        os << ", ";
        policy.IncludeTagDefinition = false;
        policy.SuppressSpecifiers = true;
      }

      Visit(*Begin);
    }
    policy = backup;
  }

  void printDeclType(clang::QualType T, llvm::StringRef DeclName, bool Pack = false) {
    if(auto tst = T->getAs<clang::TemplateSpecializationType>()){
      if(!policy.SuppressSpecifiers){
        os << tst->getAsCXXRecordDecl()->getName() << '_';
        sv.print_template_arguments(tst->template_arguments());
      }
      if(!DeclName.empty())
        os << ' ' << DeclName;
      return;
    }
    // Normally, a PackExpansionType is written as T[3]... (for instance, as a
    // template argument), but if it is the type of a declaration, the ellipsis
    // is placed before the name being declared.
    if (auto PET = T->getAs<clang::PackExpansionType>()) {
      Pack = true;
      T = PET->getPattern();
    }
    T.print(os, policy, (Pack ? "..." : "") + DeclName, indentation);
  }

  void ProcessDeclGroup(llvm::SmallVectorImpl<clang::Decl*>& Decls) {
    this->indent();
    printGroup(Decls.data(), Decls.size());
    os << ";\n";
    Decls.clear();

  }

  void Print(clang::AccessSpecifier AS) {
    switch(AS) {
    case clang::AS_none:      llvm_unreachable("No access specifier!");
    case clang::AS_public:    os << "public"; break;
    case clang::AS_protected: os << "protected"; break;
    case clang::AS_private:   os << "private"; break;
    }
  }

  void VisitDeclContext(clang::DeclContext *DC, bool indent = true, std::size_t delayed_output_layer = 0) {
    if (policy.TerseOutput)
      return;

    if (indent)
      indentation += policy.Indentation;

    llvm::SmallVector<clang::Decl*, 2> Decls;
    std::string os_source;
    llvm::raw_string_ostream ros(os_source);
    for(auto b = DC->decls_begin(), e = DC->decls_end(); b != e; ++b){
      auto&& x = *b;

      // Don't print ObjCIvarDecls, as they are printed when visiting the
      // containing ObjCInterfaceDecl.
      if (clang::isa<clang::ObjCIvarDecl>(x))
        continue;

      // Skip over implicit declarations in pretty-printing mode.
      if (x->isImplicit())
        continue;

      // Don't print implicit specializations, as they are printed when visiting
      // corresponding templates.
      if (auto FD = clang::dyn_cast<clang::FunctionDecl>(x)){
        if (FD->getTemplateSpecializationKind() == clang::TSK_ImplicitInstantiation &&
            !clang::isa<clang::ClassTemplateSpecializationDecl>(DC))
          continue;
        if ( FD->getStorageClass() == clang::SC_Static
          && FD->getReturnType().getAsString() == "void"
          && FD->getQualifiedNameAsString() == "__clpy_begin_print_out"
          && has_annotation(FD, "clpy_begin_print_out")
          && FD->param_size() == 0
          && FD->hasBody() == false){
          ++print_out_counter;
          continue;
        }
        else if ( FD->getStorageClass() == clang::SC_Static
               && FD->getReturnType().getAsString() == "void"
               && FD->getQualifiedNameAsString() == "__clpy_end_print_out"
               && has_annotation(FD, "clpy_end_print_out")
               && FD->param_size() == 0
               && FD->hasBody() == false){
          --print_out_counter;
          continue;
        }
      }

      if(print_out_counter <= 0)
        continue;

      if(delayed_outputs.size() > delayed_output_layer){
        ros << delayed_outputs.back();
        delayed_outputs.pop_back();
      }
      ros.flush();
      os << os_source;
      os_source.clear();
      auto _ = os.scoped_push(ros);

      // The next bits of code handles stuff like "struct {int x;} a,b"; we're
      // forced to merge the declarations because there's no other way to
      // refer to the struct in question.  This limited merging is safe without
      // a bunch of other checks because it only merges declarations directly
      // referring to the tag, not typedefs.
      //
      // Check whether the current declaration should be grouped with a previous
      // unnamed struct.
      auto cur_decl_type = clang::QualType{};
      if(auto tdnd = clang::dyn_cast<clang::TypedefNameDecl>(x))
        cur_decl_type = tdnd->getUnderlyingType();
      else if(auto vd = clang::dyn_cast<clang::ValueDecl>(x))
        cur_decl_type = vd->getType();
      if(!Decls.empty() && !cur_decl_type.isNull()){
        clang::QualType base_type = cur_decl_type;
        while(!base_type->isSpecifierType()){
          if(clang::isa<clang::TypedefType>(base_type))
            break;
          else if (auto pt = base_type->getAs<clang::PointerType>())
            base_type = pt->getPointeeType();
          else if (auto bpt = base_type->getAs<clang::BlockPointerType>())
            base_type = bpt->getPointeeType();
          else if (auto at = clang::dyn_cast<clang::ArrayType>(base_type))
            base_type = at->getElementType();
          else if (auto ft = base_type->getAs<clang::FunctionType>())
            base_type = ft->getReturnType();
          else if (auto vt = base_type->getAs<clang::VectorType>())
            base_type = vt->getElementType();
          else if (auto rt = base_type->getAs<clang::ReferenceType>())
            base_type = rt->getPointeeType();
          else if (auto at = base_type->getAs<clang::AutoType>())
            base_type = at->getDeducedType();
          else
            llvm_unreachable("Unknown declarator!");
        }
        if(!base_type.isNull() && clang::isa<clang::ElaboratedType>(base_type))
          base_type = clang::cast<clang::ElaboratedType>(base_type)->getNamedType();
        if (!base_type.isNull() && clang::isa<clang::TagType>(base_type) &&
            clang::cast<clang::TagType>(base_type)->getDecl() == Decls[0]) {
          Decls.push_back(x);
          continue;
        }
      }

      // If we have a merged group waiting to be handled, handle it now.
      if (!Decls.empty())
        ProcessDeclGroup(Decls);

      // If the current declaration is an unnamed tag type, save it
      // so we can merge it with the subsequent declaration(s) using it.
      if (clang::isa<clang::TagDecl>(x) && !clang::cast<clang::TagDecl>(x)->getIdentifier()) {
        Decls.push_back(x);
        continue;
      }

      if (clang::isa<clang::AccessSpecDecl>(x))
        continue;

      this->indent();
      Visit(x);

      // FIXME: Need to be able to tell the DeclPrinter when
      const char* Terminator = nullptr;
      const char* terminator_delayed = nullptr;
      if (clang::isa<clang::OMPThreadPrivateDecl>(x) || clang::isa<clang::OMPDeclareReductionDecl>(x))
        Terminator = nullptr;
      else if (clang::isa<clang::ObjCMethodDecl>(x) && clang::cast<clang::ObjCMethodDecl>(x)->hasBody())
        Terminator = nullptr;
      else if (auto FD = clang::dyn_cast<clang::FunctionDecl>(x)) {
        const bool special_definition = FD->isPure() || FD->isDefaulted() || FD->isDeleted();
        if(special_definition)
          terminator_delayed = ";\n";
        else if (FD->isThisDeclarationADefinition())
          Terminator = nullptr;
        else
          terminator_delayed = ";";
      } else if (auto TD = clang::dyn_cast<clang::FunctionTemplateDecl>(x)) {
        const bool special_definition = TD->getTemplatedDecl()->isPure() || TD->getTemplatedDecl()->isDefaulted() || TD->getTemplatedDecl()->isDeleted();
        if(special_definition)
          terminator_delayed = ";\n";
        else if (TD->getTemplatedDecl()->isThisDeclarationADefinition() && !special_definition)
          Terminator = nullptr;
        else
          terminator_delayed = ";";
      } else if (clang::isa<clang::NamespaceDecl>(x) || clang::isa<clang::LinkageSpecDecl>(x) ||
               clang::isa<clang::ObjCImplementationDecl>(x) ||
               clang::isa<clang::ObjCInterfaceDecl>(x) ||
               clang::isa<clang::ObjCProtocolDecl>(x) ||
               clang::isa<clang::ObjCCategoryImplDecl>(x) ||
               clang::isa<clang::ObjCCategoryDecl>(x))
        Terminator = nullptr;
      else if (clang::isa<clang::EnumConstantDecl>(x)) {
        if(std::next(b) != e)
          Terminator = ",";
      } else
        Terminator = ";";

      auto print_out = [&](const char* Terminator){
        if (Terminator)
          os << Terminator;
        if (!policy.TerseOutput &&
            ((clang::isa<clang::FunctionDecl>(x) &&
              clang::cast<clang::FunctionDecl>(x)->doesThisDeclarationHaveABody()) ||
             (clang::isa<clang::FunctionTemplateDecl>(x) &&
              clang::cast<clang::FunctionTemplateDecl>(x)->getTemplatedDecl()->doesThisDeclarationHaveABody())))
          ; // StmtPrinter already added '\n' after CompoundStmt.
        else
          os << '\n';
      };

      if(terminator_delayed){
        bool do_empty = delayed_outputs.empty();
        if(do_empty)
          delayed_outputs.push_back("");
        llvm::raw_string_ostream dos(delayed_outputs.back());
        if(delayed_outputs.size() == delayed_output_layer)
          os.push(dos);

        print_out(terminator_delayed);

        if(delayed_outputs.size() == delayed_output_layer)
          os.pop();
        if(do_empty)
          delayed_outputs.pop_back();
      }
      else
        print_out(Terminator);
    }

    if(delayed_outputs.size() > delayed_output_layer){
      ros << delayed_outputs.back();
      delayed_outputs.clear();
    }
    ros.flush();
    os << os_source;

    if (!Decls.empty())
      ProcessDeclGroup(Decls);

    if (indent)
      indentation -= policy.Indentation;
  }

  void VisitTranslationUnitDecl(clang::TranslationUnitDecl *D) {
    VisitDeclContext(D, false);
  }

  void VisitTypedefDecl(clang::TypedefDecl *D) {
    if (!policy.SuppressSpecifiers) {
      os << "typedef ";
      
      if (D->isModulePrivate())
        os << "__module_private__ ";
    }
    if(auto r = get_unnamed_record_decl(D->getTypeSourceInfo())){
      VisitCXXRecordDecl(r, true);
      os << D->getName();
    }
    else if(auto r = get_unnamed_enum_decl(D->getTypeSourceInfo())){
      VisitEnumDecl(r, true);
      os << D->getName();
    }
    else{
      policy.SuppressTagKeyword = 1;
      printDeclType(D->getTypeSourceInfo()->getType(), D->getName());
      policy.SuppressTagKeyword = 0;
    }
    prettyPrintAttributes(D);
  }

  void VisitTypeAliasDecl(clang::TypeAliasDecl *D) {
    os << "using " << *D;
    prettyPrintAttributes(D);
    os << " = ";
    if(auto r = get_unnamed_record_decl(D->getTypeSourceInfo()))
      VisitCXXRecordDecl(r, true);
    else if(auto r = get_unnamed_enum_decl(D->getTypeSourceInfo()))
      VisitEnumDecl(r, true);
    else
      os << D->getTypeSourceInfo()->getType().getAsString();
  }

  void VisitEnumDecl(clang::EnumDecl *D, bool force = false) {
    if(!D->isCompleteDefinition() || (!D->getIdentifier() && !force) || policy.SuppressSpecifiers)
      return;
    if (!policy.SuppressSpecifiers && D->isModulePrivate())
      os << "__module_private__ ";
    os << "enum ";
    if (D->isScoped()) {
      if (D->isScopedUsingClassTag())
        os << "class ";
      else
        os << "struct ";
    }
    os << *D;

    if (D->isFixed() && D->getASTContext().getLangOpts().CPlusPlus11)
      os << " : " << D->getIntegerType().stream(policy);

    if (D->isCompleteDefinition()) {
      os << " {\n";
      VisitDeclContext(D);
      indent() << '}';
    }
    prettyPrintAttributes(D);
  }

  void VisitEnumConstantDecl(clang::EnumConstantDecl *D) {
    os << *D;
    prettyPrintAttributes(D);
    if (clang::Expr *Init = D->getInitExpr()) {
      os << " = ";
      sv.Visit(Init);
    }
  }

  void VisitFunctionDecl(clang::FunctionDecl *D, clang::CXXMethodDecl* method = nullptr) {
    const auto annons = prettyPrintPragmas(D);
    if (!D->getDescribedFunctionTemplate() &&
        !D->isFunctionTemplateSpecialization()){
      for(auto&& x : annons) if(x == "clpy_elementwise_tag"){
        auto _ind = std::find_if(func_arg_info.back().begin(), func_arg_info.back().end(), [](const function_special_argument_info& x){
          return x.arg_flag == function_special_argument_info::cindex && x.name == "_ind";
        });
        if(_ind == func_arg_info.back().end())
          throw std::runtime_error("the function declaration annotated \"clpy_elementwise_tag\" must be in a function which has CIndexer argument named \"_ind\"");
        if(D->getReturnType().getAsString() == "void"
        && D->getQualifiedNameAsString() == "__clpy_elementwise_preprocess"
        && D->param_size() == 0
        && D->hasBody() == false){
          os << "set_CIndexer_" << _ind->ndim << "(&_ind, i);\n";
          indent() << "const size_t _ind_size = size_CIndexer_" << _ind->ndim << "(&_ind)";
        }
        else if(D->getReturnType().getAsString() == "void"
             && D->getQualifiedNameAsString() == "__clpy_elementwise_postprocess"
             && D->param_size() == 0
             && D->hasBody() == false){
          bool first = true;
          for(auto&& x : func_arg_info.back()) if(x.arg_flag == function_special_argument_info::ind && !x.is_input){
            if(first)
              first = false;
            else{
              os << ";\n";
              indent();
            }
            os << x.name << "_data[get_CArrayIndex_" << x.ndim << "(&" << x.name << "_info, &_ind)/sizeof(" << x.type << ")] = " << x.name << ";\n";
          }
        }
        return;
      }
      else if(x == "clpy_reduction_tag"){
        auto _in_ind = std::find_if(func_arg_info.back().begin(), func_arg_info.back().end(), [](const function_special_argument_info& x){
          return x.arg_flag == function_special_argument_info::cindex && x.name == "_in_ind";
        });
        if(_in_ind == func_arg_info.back().end())
          throw std::runtime_error("the function declaration annotated \"clpy_reduction_tag\" must be in a function which has CIndexer argument named \"_in_ind\"");
        auto _out_ind = std::find_if(func_arg_info.back().begin(), func_arg_info.back().end(), [](const function_special_argument_info& x){
          return x.arg_flag == function_special_argument_info::cindex && x.name == "_out_ind";
        });
        if(_out_ind == func_arg_info.back().end())
          throw std::runtime_error("the function declaration annotated \"clpy_reduction_tag\" must be in a function which has CIndexer argument named \"_out_ind\"");
        if(D->getReturnType().getAsString() == "void"
        && D->getQualifiedNameAsString() == "__clpy_reduction_preprocess"
        && D->param_size() == 0
        && D->hasBody() == false){
          os << "const size_t _in_ind_size = size_CIndexer_" << _in_ind->ndim << "(&_in_ind);\n";
          indent() << "const size_t _out_ind_size = size_CIndexer_" << _out_ind->ndim << "(&_out_ind)";
        }
        else if(D->getReturnType().getAsString() == "void"
             && D->getQualifiedNameAsString() == "__clpy_reduction_set_cindex_in"
             && D->param_size() == 0
             && D->hasBody() == false)
          os << "set_CIndexer_" << _in_ind->ndim << "(&_in_ind, _j)";
        else if(D->getReturnType().getAsString() == "void"
             && D->getQualifiedNameAsString() == "__clpy_reduction_set_cindex_out"
             && D->param_size() == 0
             && D->hasBody() == false)
          os << "set_CIndexer_" << _out_ind->ndim << "(&_out_ind, _i)";
        return;
      }
      else if(D->getReturnType().getAsString() == "void"
           && D->getQualifiedNameAsString() == "__clpy_reduction_postprocess"
           && D->param_size() == 0
           && D->hasBody() == false){
        const bool is_simple = x == "clpy_simple_reduction_tag";
        const bool is_standard = x == "clpy_standard_reduction_tag";
        if(!is_simple && !is_standard)
          continue;
        auto _in_ind = std::find_if(func_arg_info.back().begin(), func_arg_info.back().end(), [](const function_special_argument_info& x){
          return x.arg_flag == function_special_argument_info::cindex && x.name == "_in_ind";
        });
        if(_in_ind == func_arg_info.back().end())
          throw std::runtime_error("the function declaration annotated \"" + x + "\" must be in a function which has CIndexer argument named \"_in_ind\"");
        auto _out_ind = std::find_if(func_arg_info.back().begin(), func_arg_info.back().end(), [](const function_special_argument_info& x){
          return x.arg_flag == function_special_argument_info::cindex && x.name == "_out_ind";
        });
        if(_out_ind == func_arg_info.back().end())
          throw std::runtime_error("the function declaration annotated \"" + x + "\" must be in a function which has CIndexer argument named \"_out_ind\"");
        bool first = true;
        for(auto&& x : func_arg_info.back()) if(x.arg_flag == function_special_argument_info::ind && !x.is_input){
          if(first)
            first = false;
          else{
            os << ";\n";
            indent();
          }
          if(is_simple){
            const auto in_or_out = x.name.substr(0, std::find_if(x.name.begin(), x.name.end(), [](char c){return ::isdigit(c);})-x.name.begin());
            os << x.name << "_data[get_CArrayIndex_" << x.ndim << "(&" << x.name << "_info, &_" << in_or_out << "_ind)/sizeof(" << x.type << ")] = " << x.name;
          }
          else
            os << x.name << "_data[get_CArrayIndexI_" << x.ndim << "(&" << x.name << "_info, _i)/sizeof(" << x.type << ")] = " << x.name;
        }
        return;
      }
    }

    struct auto_popper{
      std::vector<std::vector<function_special_argument_info>>& f;
      auto_popper(std::vector<std::vector<function_special_argument_info>>& fsai):f{fsai}{f.emplace_back();}
      ~auto_popper(){f.pop_back();}
    }_{func_arg_info};

    clang::CXXConstructorDecl *CDecl = clang::dyn_cast<clang::CXXConstructorDecl>(D);
    clang::CXXConversionDecl *ConversionDecl = clang::dyn_cast<clang::CXXConversionDecl>(D);
    if (!policy.SuppressSpecifiers) {
      switch (D->getStorageClass()) {
      case clang::SC_None: break;
      case clang::SC_Extern: os << "extern "; break;
      case clang::SC_Static: os << "static "; break;
      case clang::SC_PrivateExtern: os << "__private_extern__ "; break;
      case clang::SC_Auto: case clang::SC_Register:
        llvm_unreachable("invalid for functions");
      }

      if (D->isInlineSpecified())  os << "inline ";
      if (D->isVirtualAsWritten()) os << "virtual ";
      if (D->isModulePrivate())    os << "__module_private__ ";
      if (D->isConstexpr() && !D->isExplicitlyDefaulted()) os << "constexpr ";
    }

    std::string Proto;
    if (!policy.SuppressScope) {
      if(auto NS = D->getQualifier()) {
        llvm::raw_string_ostream OS(Proto);
        NS->print(OS, policy);
      }
    }

    std::string parent_name;
    if(method){
      {
        llvm::raw_string_ostream Pos(parent_name);
        auto _ = os.scoped_push(Pos);
        os << method->getParent()->getNameAsString();
        if (auto S = clang::dyn_cast<clang::ClassTemplatePartialSpecializationDecl>(method->getParent())) {
          os << '_';
          sv.print_template_arguments(&S->getTemplateArgs());
        }
        else if (auto S = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(method->getParent())) {
          os << '_';
          sv.print_template_arguments(&S->getTemplateArgs());
        }
      }
      parent_name = sv.to_identifier(parent_name);
    }

    auto name = D->getNameInfo().getAsString();

    if(CDecl)
      name = "constructor";

    if(method)
      name += '_' + parent_name;

    if(auto TArgs = D->getTemplateSpecializationArgs()) {
      auto backup = policy;
      policy.SuppressSpecifiers = false;
      llvm::raw_string_ostream Pos(name);
      auto _ = os.scoped_push(Pos);
      os << '_';
      sv.print_template_arguments(TArgs);
      Pos.flush();
      policy = backup;
    }

    clang::QualType Ty = D->getType();

    if(std::find(annons.begin(), annons.end(), "clpy_no_mangle") == annons.end()){
      const bool conflicted = std::any_of(func_name.cbegin(), func_name.cend(), [&](const std::pair<clang::FunctionDecl*, std::string>& t){return t.second == name;});
      if(conflicted){
        if(auto AFT = Ty->getAs<clang::FunctionType>()) {
          const clang::FunctionProtoType *FT = nullptr;
          if (D->hasWrittenPrototype())
            FT = clang::dyn_cast<clang::FunctionProtoType>(AFT);

          name += '(';
          if(method)
            name += parent_name;
          for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
            if (!method && i) name += ", ";
            name += D->getParamDecl(i)->getType().getAsString();
          }

          if (FT && FT->isVariadic()) {
            if (D->getNumParams()) name += ", ";
            name += "...";
          }

          name += ')';
        }
        name = sv.to_identifier(name);
      }
    }
    func_name[D] = name;
    Proto += name;

    while(auto PT = clang::dyn_cast<clang::ParenType>(Ty)) {
      Proto = '(' + Proto + ')';
      Ty = PT->getInnerType();
    }

    std::string inits; // for constructor initializer list
    if(auto AFT = Ty->getAs<clang::FunctionType>()) {
      const clang::FunctionProtoType *FT = nullptr;
      if (D->hasWrittenPrototype())
        FT = clang::dyn_cast<clang::FunctionProtoType>(AFT);

      Proto += '(';
      if (FT) {
        auto backup = policy;
        policy.SuppressSpecifiers = false;
        llvm::raw_string_ostream Pos(Proto);
        auto _ = os.scoped_push(Pos);
        if(method){
          if(method->isConst())
            Pos << "const ";
          if(method->isVolatile())
            Pos << "volatile ";
          Pos << parent_name << "*const this";
        }
        else if(CDecl)
          Pos << parent_name << "*const this";
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (method || i) Pos << ", ";
          VisitParmVarDecl(D->getParamDecl(i));
        }
        backup = policy;

        if (FT->isVariadic()) {
          if (!method && D->getNumParams()) Pos << ", ";
          Pos << "...";
        }
      } else if (D->doesThisDeclarationHaveABody() && !D->hasPrototype()) {
        if(method){
          if(method->isConst())
            Proto += "const ";
          if(method->isVolatile())
            Proto += "volatile ";
          Proto += parent_name + "*const this";
        }
        else if(CDecl)
          Proto += parent_name + "*const this";
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (method || i)
            Proto += ", ";
          Proto += D->getParamDecl(i)->getNameAsString();
        }
      }

      Proto += ')';
      
      if (FT) {
        if (FT->isRestrict())
          Proto += " restrict";

        switch (FT->getRefQualifier()) {
        case clang::RQ_None:
          break;
        case clang::RQ_LValue:
        case clang::RQ_RValue:
          throw std::runtime_error("ultima doesn't support ref qualifier");
          break;
        }
      }

      auto subpolicy = policy;
      subpolicy.SuppressSpecifiers = false;
      if (CDecl) {
        llvm::raw_string_ostream ios(inits);
        auto _ = os.scoped_push(ios);
        for (const auto *BMInitializer : CDecl->inits()) {
          if (BMInitializer->isInClassMemberInitializer())
            continue;

          if (BMInitializer->isAnyMemberInitializer()) {
            clang::FieldDecl *FD = BMInitializer->getAnyMember();
            sv.Indent() << "this->" << *FD << " = ";
          } else {
            os << clang::QualType(BMInitializer->getBaseClass(), 0).getAsString(policy);
          }

          if (!BMInitializer->getInit()) {
            // Nothing to print
          } else {
            clang::Expr *Init = BMInitializer->getInit();
            if (clang::ExprWithCleanups *Tmp = clang::dyn_cast<clang::ExprWithCleanups>(Init))
              Init = Tmp->getSubExpr();
            
            Init = Init->IgnoreParens();

            clang::Expr *SimpleInit = nullptr;
            clang::Expr **Args = nullptr;
            unsigned NumArgs = 0;
            if (clang::ParenListExpr *ParenList = clang::dyn_cast<clang::ParenListExpr>(Init)) {
              Args = ParenList->getExprs();
              NumArgs = ParenList->getNumExprs();
            } else if (clang::CXXConstructExpr *Construct
                                          = clang::dyn_cast<clang::CXXConstructExpr>(Init)) {
              Args = Construct->getArgs();
              NumArgs = Construct->getNumArgs();
            } else {
              if(clang::InitListExpr* InitList = clang::dyn_cast<clang::InitListExpr>(Init)) {
                if(InitList->getNumInits() == 1 && !BMInitializer->getAnyMember()->getType().getLocalUnqualifiedType().getTypePtr()->isCompoundType())
                  Init = InitList->getInit(0);
              }
              SimpleInit = Init;
            }
            
            if (SimpleInit)
              sv.Visit(SimpleInit);
            else {
              for (unsigned I = 0; I != NumArgs; ++I) {
                assert(Args[I] != nullptr && "Expected non-null Expr");
                if (clang::isa<clang::CXXDefaultArgExpr>(Args[I]))
                  break;
                
                if (I)
                  os << ", ";
                sv.Visit(Args[I]);
              }
            }
            os << ";\n";
          }
          if (BMInitializer->isPackExpansion())
            os << "...";
        }
        Proto = "void " + Proto;
      } else if (!ConversionDecl && !clang::isa<clang::CXXDestructorDecl>(D)) {
        if (FT && FT->hasTrailingReturn()) {
          os << Proto << " -> ";
          Proto.clear();
        }
        printDeclType(AFT->getReturnType(), Proto);
        Proto.clear();
      }
      os << Proto;
    } else {
      printDeclType(Ty, Proto);
    }

    prettyPrintAttributes(D);

    if (D->isPure())
      throw std::runtime_error("ultima doesn't support pure function");
    else if (D->isDeletedAsWritten());
    else if (D->isExplicitlyDefaulted())
      os << "{}";
    else if (D->doesThisDeclarationHaveABody()) {
      if (!policy.TerseOutput) {
        if (!D->hasPrototype() && D->getNumParams()) {
          // This is a K&R function definition, so we need to print the
          // parameters.
          os << '\n';
          auto backup = policy;
          policy.SuppressSpecifiers = false;
          indentation += policy.Indentation;
          for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
            indent();
            VisitParmVarDecl(D->getParamDecl(i));
            os << ";";
          }
          policy = backup;
          indentation -= policy.Indentation;
        } else
          os << ' ';

        if (D->getBody()){
          os << '\n';
          indent() << "{\n" << inits;

          for (auto *I : clang::dyn_cast<clang::CompoundStmt>(D->getBody())->body())
            sv.PrintStmt(I);

          sv.Indent() << "}\n";
        }
      } else {
        if (clang::isa<clang::CXXConstructorDecl>(*D))
          os << " {}";
      }
    }
  }

  void VisitCXXMethodDecl(clang::CXXMethodDecl* method){
    if(delayed_outputs.empty())
      delayed_outputs.push_back("");
    llvm::raw_string_ostream dos(delayed_outputs.back());
    auto _ = os.scoped_push(dos);
    VisitFunctionDecl(clang::dyn_cast<clang::FunctionDecl>(method), method);
  }

  void VisitFriendDecl(clang::FriendDecl *D) {
    if (clang::TypeSourceInfo *TSI = D->getFriendType()) {
      unsigned NumTPLists = D->getFriendTypeNumTemplateParameterLists();
      for (unsigned i = 0; i < NumTPLists; ++i)
        printTemplateParameters(D->getFriendTypeTemplateParameterList(i));
      os << "friend ";
      os << ' ' << TSI->getType().getAsString(policy);
    }
    else if (clang::FunctionDecl *FD =
        clang::dyn_cast<clang::FunctionDecl>(D->getFriendDecl())) {
      os << "friend ";
      VisitFunctionDecl(FD);
    }
    else if (clang::FunctionTemplateDecl *FTD =
             clang::dyn_cast<clang::FunctionTemplateDecl>(D->getFriendDecl())) {
      os << "friend ";
      VisitFunctionTemplateDecl(FTD);
    }
    else if (clang::ClassTemplateDecl *CTD =
             clang::dyn_cast<clang::ClassTemplateDecl>(D->getFriendDecl())) {
      os << "friend ";
      VisitRedeclarableTemplateDecl(CTD);
    }
  }

  void VisitFieldDecl(clang::FieldDecl *D) {
    // FIXME: add printing of pragma attributes if required.
    if (!policy.SuppressSpecifiers && D->isMutable())
      os << "mutable ";
    if (!policy.SuppressSpecifiers && D->isModulePrivate())
      os << "__module_private__ ";

    if(auto r = get_unnamed_record_decl(D)){
      VisitCXXRecordDecl(r, true);
      os << ' ' << D->getName();
    }
    else if(auto r = get_unnamed_enum_decl(D->getTypeSourceInfo())){
      VisitEnumDecl(r, true);
      os << ' ' << D->getName();
    }
    else
      printDeclType(D->getType(), D->getName());

    if (D->isBitField()) {
      os << " : ";
      sv.Visit(D->getBitWidth());
    }

    clang::Expr *Init = D->getInClassInitializer();
    if (!policy.SuppressInitializers && Init) {
      if (D->getInClassInitStyle() == clang::ICIS_ListInit)
        os << ' ';
      else
        os << " = ";
      sv.Visit(Init);
    }
    prettyPrintAttributes(D);
  }

  void VisitLabelDecl(clang::LabelDecl *D) {
    os << *D << ':';
  }

  void VisitVarDecl(clang::VarDecl *D, bool parameter = false) {
    bool is_const = false;
    std::string init_str;
    bool recover_suppress_specifiers = false;
    {
      auto get_ndim = [D](clang::Expr* e){
        llvm::APSInt result;
        const auto& ast_context = D->getASTContext();
        if(e->isIntegerConstantExpr(result, ast_context))
          return result.getLimitedValue();
        else
          throw std::runtime_error("VisitVarDecl::carray_argument::get_ndim: unexpected AST");
      };
      const auto annons = prettyPrintPragmas(D);
      auto template_type = D->getType()->getAs<clang::TemplateSpecializationType>();
      auto carray_argument = [&get_ndim, this](const clang::TemplateSpecializationType* tt, const std::string& name, bool is_raw, bool is_input){
        auto val_type = tt->begin()->getAsType().getAsString(policy);
        const auto ndim = get_ndim((tt->begin()+1)->getAsExpr());
        func_arg_info.back().emplace_back(function_special_argument_info{name, val_type, is_raw ? function_special_argument_info::raw : function_special_argument_info::ind, static_cast<int>(ndim), is_input});
        //TODO: Add const to val_type when is_input is true
        os << "__global " << val_type << "* const __restrict__ " << name << (is_raw ? "" : "_data")
           << ", const CArray_" << ndim << ' ' << name << "_info";
      };
      static constexpr char clpy_arg_tag[] = "clpy_arg:";
      static constexpr char clpy_simple_reduction_tag[] = "clpy_simple_reduction_tag:";
      static constexpr char clpy_standard_reduction_tag[] = "clpy_standard_reduction_tag:";
      if(annons.empty()){
        if(parameter && template_type){
          auto template_type_name = template_type->getTemplateName().getAsTemplateDecl()->getQualifiedNameAsString();
          if(template_type_name == "CIndexer"){
            const auto ndim = get_ndim(template_type->begin()->getAsExpr());
            func_arg_info.back().emplace_back(function_special_argument_info{D->getNameAsString(), "", function_special_argument_info::cindex, static_cast<int>(ndim), true});
            os << "CIndexer_" << ndim << ' ' << D->getName();
            return;
          }
          else if(template_type_name == "CArray"){
            //TODO: Add const(is_input == true) if it should be
            carray_argument(template_type, D->getNameAsString(), true, false);
            return;
          }
        }
      }
      else for(auto&& x : annons) if(x == "clpy_ignore") return;
        else if(parameter && x.find(clpy_arg_tag) == 0){
          static constexpr std::size_t clpy_arg_tag_length = sizeof(clpy_arg_tag)-1;
          static constexpr std::size_t tag_prefix_length = clpy_arg_tag_length + 3/*"ind" or "raw"*/ + 1/*space*/;
          const bool is_raw = x.find("raw ", clpy_arg_tag_length) == clpy_arg_tag_length;
          const auto name_end = std::min(x.find(' ', tag_prefix_length), x.size());
          const auto var_name = x.substr(tag_prefix_length, name_end - tag_prefix_length);
          //const bool is_input = x.find("const", name_end) != std::string::npos;
          if(!template_type || template_type->getTemplateName().getAsTemplateDecl()->getQualifiedNameAsString() != "CArray")
            throw std::runtime_error("invalid \"clpy_arg\" annotation (it is only for CArray argument)");
          //TODO: Add const(is_input == true) if it should be
          carray_argument(template_type, var_name, is_raw, false);
          return;
        }
        else if(!parameter && x == "clpy_elementwise_tag"){
          auto var_info = std::find_if(func_arg_info.back().begin(), func_arg_info.back().end(), [D](const function_special_argument_info& t){
            return t.name == D->getName();
          });
          if(var_info == func_arg_info.back().end())
            throw std::runtime_error("invalid \"clpy_elementwise_tag\" annotation (there is no related argument)");
          is_const = var_info->is_input;
          init_str = " = " + var_info->name + "_data[get_CArrayIndex_" + std::to_string(var_info->ndim) + "(&" + var_info->name + "_info, &_ind)/sizeof(" + var_info->type + ")]";
        }
        else if(!parameter && x.find(clpy_simple_reduction_tag) == 0){
          static constexpr std::size_t tag_length = sizeof(clpy_simple_reduction_tag)-1;
          auto var_info = std::find_if(func_arg_info.back().begin(), func_arg_info.back().end(), [D](const function_special_argument_info& t){
            return t.name == D->getName();
          });
          if(var_info == func_arg_info.back().end())
            throw std::runtime_error("invalid \"clpy_simple_reduction_tag\" annotation (there is no related argument)");
          const auto& name = var_info->name;
          is_const = var_info->is_input;
          init_str = " = " + name + "_data[get_CArrayIndex_" + std::to_string(var_info->ndim) + "(&" + name + "_info, &_" + x.substr(tag_length) + "_ind)/sizeof(" + var_info->type + ")]";
          if(x.substr(tag_length) == "in"){
            policy.SuppressSpecifiers = true;
            recover_suppress_specifiers = true;
          }
        }
        else if(!parameter && x.find(clpy_standard_reduction_tag) == 0){
          static constexpr std::size_t tag_length = sizeof(clpy_standard_reduction_tag)-1;
          auto var_info = std::find_if(func_arg_info.back().begin(), func_arg_info.back().end(), [D](const function_special_argument_info& t){
            return t.name == D->getName();
          });
          if(var_info == func_arg_info.back().end())
            throw std::runtime_error("invalid \"clpy_simple_reduction_tag\" annotation (there is no related argument)");
          const auto& name = var_info->name;
          is_const = var_info->is_input;
          init_str = " = " + name + "_data[get_CArrayIndexI_" + std::to_string(var_info->ndim) + "(&" + name + "_info, _" + x.substr(tag_length) + ")/sizeof(" + var_info->type + ")]";
          if(x.substr(tag_length) == "j"){
            policy.SuppressSpecifiers = true;
            recover_suppress_specifiers = true;
          }
        }
    }

    clang::QualType T = D->getTypeSourceInfo()
      ? D->getTypeSourceInfo()->getType()
      : D->getASTContext().getUnqualifiedObjCPointerType(D->getType());

    if (!policy.SuppressSpecifiers) {
      clang::StorageClass SC = D->getStorageClass();
      if (SC != clang::SC_None)
        os << clang::VarDecl::getStorageClassSpecifierString(SC) << ' ';

      switch (D->getTSCSpec()) {
      case clang::TSCS_unspecified:
        break;
      case clang::TSCS___thread:
        os << "__thread ";
        break;
      case clang::TSCS__Thread_local:
        os << "_Thread_local ";
        break;
      case clang::TSCS_thread_local:
        os << "thread_local ";
        break;
      }

      if (D->isModulePrivate())
        os << "__module_private__ ";

      if (D->isConstexpr()) {
        os << "constexpr ";
        T.removeLocalConst();
      }
    }

    if(is_const)
      os << "const ";

    if(auto r = get_unnamed_record_decl(D)){
      VisitCXXRecordDecl(r, true);
      os << ' ' << D->getName();
    }
    else if(auto r = get_unnamed_enum_decl(D->getTypeSourceInfo())){
      VisitEnumDecl(r, true);
      os << ' ' << D->getName();
    }
    else
      printDeclType(T, D->getName());
    clang::Expr *Init = D->getInit();
    auto dig_elidable = [](clang::CXXConstructExpr* E){
      while(E && E->isElidable()){
        auto a = E->getArg(0);
        if(auto subexpr = clang::dyn_cast<clang::CXXConstructExpr>(a->IgnoreImplicit()))
          E = subexpr;
        else
          break;
      }
      return E;
    };
    auto Construct = Init ? dig_elidable(clang::dyn_cast<clang::CXXConstructExpr>(Init->IgnoreImplicit())) : nullptr;
    if (!policy.SuppressInitializers && Init) {
      bool ImplicitInit = false;
      if (Construct) {
        if (D->getInitStyle() == clang::VarDecl::CallInit &&
            !Construct->isListInitialization()) {
          ImplicitInit = Construct->getNumArgs() == 0 ||
            Construct->getArg(0)->isDefaultArgument();
        }
      }
      if (!ImplicitInit) {
        if ((D->getInitStyle() == clang::VarDecl::CInit || D->getInitStyle() == clang::VarDecl::ListInit) && (!Construct || Construct->getConstructor()->isDefaulted()))
          os << " = ";
        else
          os << ';';
        auto backup = policy;
        policy.SuppressSpecifiers = false;
        policy.IncludeTagDefinition = false;
        if(Construct)
          sv.VisitCXXConstructExpr(Construct, D->getNameAsString().c_str());
        else
          sv.Visit(Init);
        policy = backup;
      }
      else if(Construct){
        const bool needs_equal = (D->getInitStyle() == clang::VarDecl::CInit || D->getInitStyle() == clang::VarDecl::ListInit) && Construct->getConstructor()->isDefaulted();
        if(needs_equal)
          os << " = ";
        std::string str;
        llvm::raw_string_ostream ss(str);
        os.push(ss);
        sv.VisitCXXConstructExpr(Construct, D->getNameAsString().c_str());
        os.pop();
        ss.flush();
        if(!str.empty() && !needs_equal){
          llvm::raw_string_ostream ros{delayed_outputs.back()};
          ros << str << ';';
        }
        else
          os << str;
      }
    }
    else if(!init_str.empty())
      os << init_str;
    prettyPrintAttributes(D);
    if(recover_suppress_specifiers)
      policy.SuppressSpecifiers = false;
  }

  void VisitParmVarDecl(clang::ParmVarDecl *D) {
    VisitVarDecl(D, true);
  }

  void VisitFileScopeAsmDecl(clang::FileScopeAsmDecl *D) {
    os << "__asm (";
    sv.Visit(D->getAsmString());
    os << ')';
  }

  void VisitStaticAssertDecl(clang::StaticAssertDecl *D) {
    os << "static_assert(";
    sv.Visit(D->getAssertExpr());
    if (clang::StringLiteral *SL = D->getMessage()) {
      os << ", ";
      sv.Visit(SL);
    }
    os << ')';
  }

  void VisitNamespaceDecl(clang::NamespaceDecl *D) {
    if (D->isInline())
      os << "inline ";
    os << "namespace " << *D << " {\n";
    VisitDeclContext(D);
    indent() << '}';
  }

  void VisitUsingDirectiveDecl(clang::UsingDirectiveDecl *D) {
    os << "using namespace ";
    if (D->getQualifier())
      D->getQualifier()->print(os, policy);
    os << *D->getNominatedNamespaceAsWritten();
  }

  void VisitNamespaceAliasDecl(clang::NamespaceAliasDecl *D) {
    os << "namespace " << *D << " = ";
    if (D->getQualifier())
      D->getQualifier()->print(os, policy);
    os << *D->getAliasedNamespace();
  }

  void VisitEmptyDecl(clang::EmptyDecl *D) {
    prettyPrintAttributes(D);
  }

  void VisitCXXRecordDecl(clang::CXXRecordDecl *D, bool force = false) {
    if(!D->isCompleteDefinition() || (!D->getIdentifier() && !force) || policy.SuppressSpecifiers)
      return;
    if(D->getKind() == clang::Decl::Kind::Enum){
      D->print(os, indentation);
    }
    // FIXME: add printing of pragma attributes if required.
    if (!policy.SuppressSpecifiers && D->isModulePrivate())
      os << "__module_private__ ";
    if(!force)
      os << "typedef ";
    if(D->getKindName() == "class")
      os << "struct";
    else
      os << D->getKindName();

    prettyPrintAttributes(D);

    std::string name;

    if (D->getIdentifier()) {
      {
        llvm::raw_string_ostream nos(name);
        auto _ = os.scoped_push(nos);
        os << *D;

        if (auto S = clang::dyn_cast<clang::ClassTemplatePartialSpecializationDecl>(D)) {
          os << '_';
          sv.print_template_arguments(&S->getTemplateArgs());
        }
        else if (auto S = clang::dyn_cast<clang::ClassTemplateSpecializationDecl>(D)) {
          os << '_';
          sv.print_template_arguments(&S->getTemplateArgs());
        }
      }
      name = sv.to_identifier(name);
      os << ' ' << name;
    }

    if (D->isCompleteDefinition()) {
      // Print the base classes
      if (D->getNumBases()) {
        os << " : ";
        for (auto Base = D->bases_begin(),
               BaseEnd = D->bases_end(); Base != BaseEnd; ++Base) {
          if (Base != D->bases_begin())
            os << ", ";

          if (Base->isVirtual())
            os << "virtual ";

          clang::AccessSpecifier AS = Base->getAccessSpecifierAsWritten();
          if (AS != clang::AS_none) {
            Print(AS);
            os << ' ';
          }
          os << Base->getType().getAsString(policy);

          if (Base->isPackExpansion())
            os << "...";
        }
      }

      // Print the class definition
      if (policy.TerseOutput) {
        os << " {}";
      } else {
        os << " {\n";
        delayed_outputs.push_back("");
        VisitDeclContext(D, true, delayed_outputs.size());
        indent() << '}';
        if(D->getIdentifier()){
          os << name;
        }
      }
    }
  }

  void VisitLinkageSpecDecl(clang::LinkageSpecDecl *D) {
    const char *l;
    if (D->getLanguage() == clang::LinkageSpecDecl::lang_c)
      l = "C";
    else {
      assert(D->getLanguage() == clang::LinkageSpecDecl::lang_cxx &&
             "unknown language in linkage specification");
      l = "C++";
    }

    os << "extern \"" << l << "\" ";
    if (D->hasBraces()) {
      os << "{\n";
      VisitDeclContext(D);
      indent() << '}';
    } else
      Visit(*D->decls_begin());
  }

  void printTemplateParameters(const clang::TemplateParameterList *Params) {
    assert(Params);

    os << "template <";

    for (unsigned i = 0, e = Params->size(); i != e; ++i) {
      if (i != 0)
        os << ", ";

      const auto* Param = Params->getParam(i);
      if (auto TTP = clang::dyn_cast<clang::TemplateTypeParmDecl>(Param)) {

        if (TTP->wasDeclaredWithTypename())
          os << "typename ";
        else
          os << "class ";

        if (TTP->isParameterPack())
          os << "...";

        os << *TTP;

        if (TTP->hasDefaultArgument()) {
          os << " = ";
          os << TTP->getDefaultArgument().getAsString(policy);
        };
      } else if (auto NTTP = clang::dyn_cast<clang::NonTypeTemplateParmDecl>(Param)) {
        llvm::StringRef Name;
        if (clang::IdentifierInfo *II = NTTP->getIdentifier())
          Name = II->getName();
        printDeclType(NTTP->getType(), Name, NTTP->isParameterPack());

        if (NTTP->hasDefaultArgument()) {
          os << " = ";
          sv.Visit(NTTP->getDefaultArgument());
        }
      } else if (auto TTPD = clang::dyn_cast<clang::TemplateTemplateParmDecl>(Param)) {
        VisitTemplateDecl(TTPD);
        // FIXME: print the default argument, if present.
      }
    }

    os << "> ";
  }

  void printTemplateArguments(const clang::TemplateArgumentList &Args,
                              const clang::TemplateParameterList *Params = nullptr) {
    os << '<';
    for (size_t I = 0, E = Args.size(); I < E; ++I) {
      const clang::TemplateArgument &A = Args[I];
      if (I)
        os << ", ";
      if (Params) {
        if (A.getKind() == clang::TemplateArgument::Type)
          if (auto T = A.getAsType()->getAs<clang::TemplateTypeParmType>()) {
            auto P = clang::cast<clang::TemplateTypeParmDecl>(Params->getParam(T->getIndex()));
            os << *P;
            continue;
          }
        if (A.getKind() == clang::TemplateArgument::Template) {
          if (auto T = A.getAsTemplate().getAsTemplateDecl())
            if (auto TD = clang::dyn_cast<clang::TemplateTemplateParmDecl>(T)) {
              auto P = clang::cast<clang::TemplateTemplateParmDecl>(
                                                Params->getParam(TD->getIndex()));
              os << *P;
              continue;
            }
        }
        if (A.getKind() == clang::TemplateArgument::Expression) {
          if (auto E = clang::dyn_cast<clang::DeclRefExpr>(A.getAsExpr()))
            if (auto N = clang::dyn_cast<clang::NonTypeTemplateParmDecl>(E->getDecl())) {
              auto P = clang::cast<clang::NonTypeTemplateParmDecl>(
                                                 Params->getParam(N->getIndex()));
              os << *P;
              continue;
            }
        }
      }
      A.print(policy, os);
    }
    os << '>';
  }

  void VisitTemplateDecl(const clang::TemplateDecl *D) {
    printTemplateParameters(D->getTemplateParameters());

    if(auto TTP = clang::dyn_cast<clang::TemplateTemplateParmDecl>(D)){
      os << "class ";
      if (TTP->isParameterPack())
        os << "...";
      os << D->getName();
    } else {
      Visit(D->getTemplatedDecl());
    }
  }

  void VisitFunctionTemplateDecl(clang::FunctionTemplateDecl *D) {
    bool first = true;
    std::string str;
    llvm::raw_string_ostream ros(str);
    for(auto&& x : D->specializations()){
      auto _ = os.scoped_push(ros);
      if(first)
        first = false;
      else{
        indent();
        os << '\n';
      }
      Visit(x);
      ros.flush();
      *os.oss.front() << str;
      str.clear();
    }
  }

  void VisitClassTemplateDecl(clang::ClassTemplateDecl *D) {
    for (auto *I : D->specializations()){
      if (D->isThisDeclarationADefinition())
        os << ';';
      os << '\n';
      Visit(I);
    }
  }

  void VisitClassTemplateSpecializationDecl(clang::ClassTemplateSpecializationDecl *D) {
    VisitCXXRecordDecl(D);
  }

  void VisitClassTemplatePartialSpecializationDecl(clang::ClassTemplatePartialSpecializationDecl*) {}

  void VisitUsingDecl(clang::UsingDecl *D) {
    if (!D->isAccessDeclaration())
      os << "using ";
    if (D->hasTypename())
      os << "typename ";
    D->getQualifier()->print(os, policy);

    // Use the correct record name when the using declaration is used for
    // inheriting constructors.
    for (const auto *Shadow : D->shadows()) {
      if(const auto *ConstructorShadow =
              clang::dyn_cast<clang::ConstructorUsingShadowDecl>(Shadow)) {
        assert(Shadow->getDeclContext() == ConstructorShadow->getDeclContext());
        os << *ConstructorShadow->getNominatedBaseClass();
        return;
      }
    }
    os << *D;
  }

  void
  VisitUnresolvedUsingTypenameDecl(clang::UnresolvedUsingTypenameDecl *D) {
    os << "using typename ";
    D->getQualifier()->print(os, policy);
    os << D->getDeclName();
  }

  void VisitUnresolvedUsingValueDecl(clang::UnresolvedUsingValueDecl *D) {
    if (!D->isAccessDeclaration())
      os << "using ";
    D->getQualifier()->print(os, policy);
    os << D->getDeclName();
  }

  void VisitUsingShadowDecl(clang::UsingShadowDecl*) {}
};

namespace registrar{

class ast_consumer : public clang::ASTConsumer{
  std::unique_ptr<decl_visitor> visit;
  static clang::PrintingPolicy ppolicy(clang::PrintingPolicy pp){
    pp.Bool = true;
    return pp;
  }
 public:
  explicit ast_consumer(clang::CompilerInstance& ci) : visit{new decl_visitor{llvm::outs(), ppolicy(ci.getASTContext().getPrintingPolicy())}}{
    ci.getPreprocessor().addPPCallbacks(std::make_unique<preprocessor>());
  }
  virtual void HandleTranslationUnit(clang::ASTContext& context)override{
    visit->Visit(context.getTranslationUnitDecl());
  }
};

struct ast_frontend_action : clang::SyntaxOnlyAction{
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& ci, clang::StringRef)override{
    return std::make_unique<ast_consumer>(ci);
  }
};

}

}

int main(int argc, const char** argv){
  llvm::cl::OptionCategory tool_category("ultima options");
  llvm::cl::extrahelp common_help(clang::tooling::CommonOptionsParser::HelpMessage);
  std::vector<const char*> params;
  params.reserve(argc+1);
  std::copy(argv, argv+argc, std::back_inserter(params));
  params.emplace_back("-D__ULTIMA=1");
  params.emplace_back("-xc++");
  params.emplace_back("-std=c++14");
  params.emplace_back("-w");
  params.emplace_back("-Wno-narrowing");
  params.emplace_back("-includecl_stub.hpp");
  params.emplace_back("-includecuda_stub.hpp");
  clang::tooling::CommonOptionsParser options_parser(argc = static_cast<int>(params.size()), params.data(), tool_category);
  clang::tooling::ClangTool tool(options_parser.getCompilations(), options_parser.getSourcePathList());
  return tool.run(clang::tooling::newFrontendActionFactory<ultima::registrar::ast_frontend_action>().get());
}
