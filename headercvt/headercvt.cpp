#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <utility>
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

namespace headercvt{

class MarkableLLVMOsOstream{
  llvm::raw_os_ostream llvm_ost;
  bool used = false;
public:
  MarkableLLVMOsOstream(std::ofstream& ofstream):llvm_ost(ofstream){
  }
  template <typename T>
  MarkableLLVMOsOstream& operator << (T&& rhs){
    llvm_ost << std::forward<T>(rhs);
    used = true;
    return *this;
  }
  decltype(llvm_ost)& without_using(){
    return llvm_ost;
  }
  void use(){
    used = true;
  }
  bool vacant() const{
    return !used;
  }
  void flush(){
    llvm_ost.flush();
  }
};

std::ofstream
  types("types.pxi"),
  func_decl("func_decl.pxi"),
  preprocessor_defines("preprocessor_defines.pxi"),
  not_handled("not_handled.txt");
MarkableLLVMOsOstream
  types_ostream(types),
  func_decl_ostream(func_decl),
  preprocessor_defines_ostream(preprocessor_defines),
  not_handled_ostream(not_handled);
unsigned types_indentation = 0, func_decl_indentation = 0, preprocessor_defines_indentation = 0;

static constexpr char const* indent_str = "    ";

static std::string clean_type_string(clang::QualType const& type){

  if (auto attrtype_ptr = clang::dyn_cast<clang::VectorType>(type)){
    return attrtype_ptr->getElementType().getAsString();
  }

  if (auto ptrtype_ptr = clang::dyn_cast<clang::PointerType>(type)){
    if (auto elaboratedtype_ptr = clang::dyn_cast<clang::ElaboratedType>(ptrtype_ptr->getPointeeType())){
      if (auto recordtype_ptr = clang::dyn_cast<clang::RecordType>(elaboratedtype_ptr->desugar())){
        return recordtype_ptr -> getDecl() -> getNameAsString() + " *";
      }
    }
  }

  return type.getAsString();
}


class preprocessor_defines_extractor : public clang::PPCallbacks{
private:
  unsigned Indentation;
  MarkableLLVMOsOstream& Out;

  MarkableLLVMOsOstream& Indent() { return Indent(Indentation); }
  MarkableLLVMOsOstream& Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out << indent_str;
    return Out;
  }

public:
  preprocessor_defines_extractor(MarkableLLVMOsOstream& Out): Out(Out){
    Indentation = preprocessor_defines_indentation;
  }

  void MacroDefined(const clang::Token& MacroNameTok, const clang::MacroDirective *MD) override{
    const clang::MacroDirective::Kind kind = MD->getKind();
    if (!(kind == clang::MacroDirective::Kind::MD_Define))
      return;

    const auto identifier = MacroNameTok.getIdentifierInfo()->getName().str();
    const std::regex cl_macro_detector(R"(CL_.*)");
    if (!std::regex_match(identifier, cl_macro_detector))
      return;
    Indent() << identifier << "\n";
  }
};



  static clang::QualType getDeclType(clang::Decl* D) {
    if (clang::TypedefNameDecl* TDD = clang::dyn_cast<clang::TypedefNameDecl>(D))
      return TDD->getUnderlyingType();
    if (clang::ValueDecl* VD = clang::dyn_cast<clang::ValueDecl>(D))
      return VD->getType();
    return clang::QualType();
  }

  static clang::QualType GetBaseType(clang::QualType T) {
    clang::QualType BaseType = T;
    while (!BaseType->isSpecifierType()) {
      if (const clang::PointerType *PTy = BaseType->getAs<clang::PointerType>())
        BaseType = PTy->getPointeeType();
      else if (const clang::BlockPointerType *BPy = BaseType->getAs<clang::BlockPointerType>())
        BaseType = BPy->getPointeeType();
      else if (const clang::ArrayType* ATy = clang::dyn_cast<clang::ArrayType>(BaseType))
        BaseType = ATy->getElementType();
      else if (const clang::FunctionType* FTy = BaseType->getAs<clang::FunctionType>())
        BaseType = FTy->getReturnType();
      else if (const clang::VectorType *VTy = BaseType->getAs<clang::VectorType>())
        BaseType = VTy->getElementType();
      else if (const clang::ReferenceType *RTy = BaseType->getAs<clang::ReferenceType>())
        BaseType = RTy->getPointeeType();
      else if (const clang::AutoType *ATy = BaseType->getAs<clang::AutoType>())
        BaseType = ATy->getDeducedType();
      else if (const clang::ParenType *PTy = BaseType->getAs<clang::ParenType>())
        BaseType = PTy->desugar();
      else
        // This must be a syntax error.
        break;
    }
    return BaseType;
  }



class simple_vardecl_printer : public clang::DeclVisitor<simple_vardecl_printer>{
  llvm::raw_ostream &Out;
  clang::PrintingPolicy Policy;
  unsigned Indentation;
public:
  simple_vardecl_printer(llvm::raw_ostream &Out, const clang::PrintingPolicy &Policy,
      const clang::ASTContext &, unsigned Indentation = 0)
    : Out(Out), Policy(Policy), Indentation(Indentation)
  {}

  void VisitParmVarDecl(clang::ParmVarDecl *D) {
    VisitVarDecl(D);
  }
  void VisitVarDecl(clang::VarDecl *D) {
    clang::QualType T = D->getTypeSourceInfo()
      ? D->getTypeSourceInfo()->getType()
      : D->getASTContext().getUnqualifiedObjCPointerType(D->getType());

    printDeclType(T, D->getName());
  }
  void printDeclType(clang::QualType T, llvm::StringRef DeclName, bool=false) {
    T.print(Out, Policy, DeclName, Indentation);
  }
};


class typedef_printer : public clang::DeclVisitor<typedef_printer>{
  MarkableLLVMOsOstream &Out;
  clang::PrintingPolicy Policy;
  unsigned Indentation;

  MarkableLLVMOsOstream& Indent() { return Indent(Indentation); }
  MarkableLLVMOsOstream& Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out << indent_str;
    return Out;
  }
public:
  typedef_printer(MarkableLLVMOsOstream &Out, const clang::PrintingPolicy &Policy,
              const clang::ASTContext &, unsigned Indentation = 0)
      : Out(Out), Policy(Policy), Indentation(Indentation) {}


  void VisitTypedefDecl(clang::TypedefDecl* D) {
    if (D->getTypeSourceInfo()->getType().getTypePtr()->isUnionType()){
      Out.without_using() << "# union reference ignored\n";
      return;
    }

    if (D->getTypeSourceInfo()->getType()->isFunctionPointerType()){
      Out.without_using() << "# function pointer ignored\n";
      return;
    }

    Indent();
    if (!Policy.SuppressSpecifiers) {
      Out << "ctypedef ";
    }

    clang::PrintingPolicy SubPolicy(Policy);
    SubPolicy.PolishForDeclaration = 1;
    auto Ty = D->getTypeSourceInfo()->getType().getUnqualifiedType();

    Out << clean_type_string(Ty);

    Out << " " << D->getName();
    Out << "\n";
  }

  void VisitRecordDecl(clang::RecordDecl *D) {
    if (D->isUnion()) {
      Out.without_using() << "# union declaration(" << D->getName() << ") ignored\n";
      return;
    }
    const std::regex pthread_detector(R"(.*pthread.*)");
    if(std::regex_match(D->getNameAsString(), pthread_detector)) {
      Out.without_using() << "# pthread-related type ignored\n";
      return;
    }

    Indent();
    Out << "cdef " << D->getKindName() << " " << D->getName() << ":\n";
    Indentation++;

    if( D->field_empty() ){
      Indent() << "pass\n";
    }else{
      for(auto itr = D->field_begin(); itr != D->field_end(); ++itr){
        VisitFieldDecl(*itr);
        Out << "\n";
      }
    }
  }

  void VisitFieldDecl(clang::FieldDecl *D) {
    Indent()
      << D->getASTContext().getUnqualifiedObjCPointerType(D->getType()).
      stream(Policy, D->getName(), Indentation);
  }

};




class grouped_typedef_struct_printer : public clang::DeclVisitor<grouped_typedef_struct_printer>{
  MarkableLLVMOsOstream &Out;
  clang::PrintingPolicy Policy;
  unsigned Indentation;

  MarkableLLVMOsOstream& Indent() { return Indent(Indentation); }
  MarkableLLVMOsOstream& Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out << indent_str;
    return Out;
  }

public:
  grouped_typedef_struct_printer(MarkableLLVMOsOstream &Out, const clang::PrintingPolicy &Policy,
      const clang::ASTContext &, unsigned Indentation = 0)
    : Out(Out), Policy(Policy), Indentation(Indentation) {
    }

  void visit_group_struct_decl(clang::RecordDecl *RD, clang::TypedefDecl *TDD){
    const std::regex pthread_detector(R"(.*pthread.*)");
    if(std::regex_match(RD->getNameAsString(), pthread_detector)
        || std::regex_match(TDD->getNameAsString(), pthread_detector)) {
      Out.without_using() << "# pthread-related type ignored\n";
      return;
    }



    if(!RD->isCompleteDefinition()){
      Indent()
        << "cdef struct "
        << RD->getName()
        << ":\n"
        ;
      Indentation++;
      Indent() << "pass\n";
      Indentation--;
      Indent()
        << "ctypedef "
        << clean_type_string(TDD->getTypeSourceInfo()->getType())
        << " "
        << TDD->getName()
        << "\n";
    }
    else{
      Indent()
        << "ctypedef struct "
        << TDD->getName()
        << ":\n"
        ;
      Indentation++;
      if( RD->field_empty() ){
        Indent() << "pass\n";
      }else{
        for(auto itr = RD->field_begin(); itr != RD->field_end(); ++itr){
          VisitFieldDecl(*itr);
          Out << "\n";
        }
      }
    }
  }
  void VisitFieldDecl(clang::FieldDecl *D) {
    if (D->isAnonymousStructOrUnion()){
      return;
    }

    Indent()
      << D->getASTContext().getUnqualifiedObjCPointerType(D->getType()).
      stream(Policy, D->getName(), Indentation);
  }

};

class funcdecl_printer : public clang::DeclVisitor<funcdecl_printer>{
  MarkableLLVMOsOstream &Out;
  clang::PrintingPolicy Policy;
  const clang::ASTContext &Context;
  unsigned Indentation;

  MarkableLLVMOsOstream& Indent() { return Indent(Indentation); }
  MarkableLLVMOsOstream& Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out << indent_str;
    return Out;
  }

public:
  funcdecl_printer(MarkableLLVMOsOstream &Out, const clang::PrintingPolicy &Policy,
              const clang::ASTContext &Context, unsigned Indentation = 0)
      : Out(Out), Policy(Policy), Context(Context), Indentation(Indentation) {}



  void VisitFunctionDecl(clang::FunctionDecl *D) {
    auto const function_name = D->getNameInfo().getAsString();
    const std::regex cl_function_detector(R"(cl[A-Z].*)");
    if (!std::regex_match(function_name, cl_function_detector))
      return;

    clang::PrintingPolicy SubPolicy(Policy);
    SubPolicy.SuppressSpecifiers = false;
    std::string Proto;
    Indent();

    Proto += D->getNameInfo().getAsString();

    clang::QualType Ty = D->getType();
    while (const clang::ParenType *PT = clang::dyn_cast<clang::ParenType>(Ty)) {
      Proto = '(' + Proto + ')';
      Ty = PT->getInnerType();
    }

    if (const clang::FunctionType *AFT = Ty->getAs<clang::FunctionType>()) {
      const clang::FunctionProtoType *FT = nullptr;
      if (D->hasWrittenPrototype())
        FT = clang::dyn_cast<clang::FunctionProtoType>(AFT);

      Proto += "(";
      if (FT) {
        llvm::raw_string_ostream POut(Proto);
        simple_vardecl_printer ParamPrinter(POut, SubPolicy, Context, Indentation);

        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i) POut << ", ";
          ParamPrinter.VisitParmVarDecl(D->getParamDecl(i));
        }

      } else if (D->doesThisDeclarationHaveABody() && !D->hasPrototype()) {
        for (unsigned i = 0, e = D->getNumParams(); i != e; ++i) {
          if (i)
            Proto += ", ";
          Proto += D->getParamDecl(i)->getNameAsString();
        }
      }

      Proto += ")";


      AFT->getReturnType().print(Out.without_using(), Policy, Proto);
      Proto.clear();
      Out << Proto;
    } else {
      Ty.print(Out.without_using(), Policy, Proto);
      Out.use();
    }

    Out << "\n";
  }

};


class general_decl_visitor : public clang::DeclVisitor<general_decl_visitor>{
  MarkableLLVMOsOstream &Out;
  clang::PrintingPolicy Policy;
  const clang::ASTContext &Context;
  unsigned Indentation;

  MarkableLLVMOsOstream& Indent() { return Indent(Indentation); }
  MarkableLLVMOsOstream& Indent(unsigned Indentation) {
    for (unsigned i = 0; i != Indentation; ++i)
      Out << indent_str;
    return Out;
  }

public:
  general_decl_visitor(MarkableLLVMOsOstream &Out, const clang::PrintingPolicy &Policy,
      const clang::ASTContext &Context, unsigned Indentation = 0)
    : Out(Out), Policy(Policy), Context(Context), Indentation(Indentation) { }

  void VisitTranslationUnitDecl(clang::TranslationUnitDecl *D) {
    VisitDeclContext(D, false);
  }
  void VisitDeclContext(clang::DeclContext *DC, bool Indent) {
    if (Policy.TerseOutput)
      return;

    if (Indent)
      Indentation += Policy.Indentation;

    clang::SmallVector<clang::Decl*, 2> Decls;
    for (clang::DeclContext::decl_iterator D = DC->decls_begin(), DEnd = DC->decls_end();
        D != DEnd; ++D) {

      if (D->isImplicit())
        continue;

      // The next bits of code handle stuff like "struct {int x;} a,b"; we're
      // forced to merge the declarations because there's no other way to
      // refer to the struct in question.  When that struct is named instead, we
      // also need to merge to avoid splitting off a stand-alone struct
      // declaration that produces the warning ext_no_declarators in some
      // contexts.
      //
      // This limited merging is safe without a bunch of other checks because it
      // only merges declarations directly referring to the tag, not typedefs.
      //
      // Check whether the current declaration should be grouped with a previous
      // non-free-standing tag declaration.
      clang::QualType CurDeclType = getDeclType(*D);
      if (!Decls.empty() && !CurDeclType.isNull()) {
        clang::QualType BaseType = GetBaseType(CurDeclType);
        if (!BaseType.isNull() && clang::isa<clang::ElaboratedType>(BaseType)){
          BaseType = clang::cast<clang::ElaboratedType>(BaseType)->getNamedType();
        }
        if (!BaseType.isNull() && clang::isa<clang::TagType>(BaseType) &&
            clang::cast<clang::TagType>(BaseType)->getDecl() == Decls[0]) {
          Decls.push_back(*D);
          continue;
        }
      }

      // If we have a merged group waiting to be handled, handle it now.
      if (!Decls.empty())
        ProcessDeclGroup(Decls);

      // If the current declaration is not a free standing declaration, save it
      // so we can merge it with the subsequent declaration(s) using it.
      if (clang::isa<clang::TagDecl>(*D) && !clang::cast<clang::TagDecl>(*D)->isFreeStanding()) {
        Decls.push_back(*D);
        continue;
      }


      this->Indent();
      if (clang::isa<clang::FunctionDecl>(*D)){
        funcdecl_printer FuncDeclPrinter(func_decl_ostream, Policy, Context, func_decl_indentation);
        FuncDeclPrinter.Visit(clang::cast<clang::FunctionDecl>(*D));
      }

      if (clang::isa<clang::TypedefDecl>(*D) || clang::isa<clang::RecordDecl>(*D)){
        typedef_printer TypedefPrinter(types_ostream, Policy, Context, types_indentation);
        TypedefPrinter.Visit(*D);
      }

    } // end of in-declcontext iteration

    if (!Decls.empty())
      ProcessDeclGroup(Decls);

    if (Indent)
      Indentation -= Policy.Indentation;
  }

  void ProcessDeclGroup(clang::SmallVectorImpl<clang::Decl*>& Decls) {
    this->Indent();
    processTypedefStructDeclGroup(Decls.data(), Decls.size(), types_ostream);
    Decls.clear();
  }

  void processTypedefStructDeclGroup(clang::Decl** Begin, unsigned NumDecls, MarkableLLVMOsOstream &Out) {
    if(NumDecls == 1) return;

    clang::TagDecl* TD = clang::dyn_cast<clang::TagDecl>(*Begin);
    clang::TypedefDecl* TDD = clang::dyn_cast<clang::TypedefDecl>(*(Begin+1));
    if (TD && TDD){
      clang::TagTypeKind const kind = TD -> getTagKind();
      switch (kind){
        case clang::TagTypeKind::TTK_Struct:
          // Group is a `typedef struct {field_name, ...} type_name;` declaration group
/*
|-RecordDecl struct definition      <- Begin, TD
| `-FieldDecl field_name 'int [2]'
|-TypedefDecl referenced type_name 'struct type_name':'type_name'  <- TDD
| `-ElaboratedType 'struct type_name' sugar
|   `-RecordType 'type_name'
|     `-Record ''
*/
          // or `typedef _hoge* hoge_t;` (w/o declaring _hoge) declaration group
/*
|-RecordDecl struct _hoge               <- Begin, TD
|-TypedefDecl hoge_t 'struct _hoge *'   <- TDD
| `-PointerType 'struct _hoge *'
|   `-ElaboratedType 'struct _hoge' sugar
|     `-RecordType 'struct _hoge'
|       `-Record '_hoge'
*/
          {
            auto tdp = grouped_typedef_struct_printer(Out, Policy, Context, types_indentation);
            clang::RecordDecl* RD = clang::dyn_cast<clang::RecordDecl>(TD);
            tdp.visit_group_struct_decl( RD, TDD );
            return;
          }
        case clang::TagTypeKind::TTK_Union :
          Out.without_using() << "# union declaration ignored\n";
          return;
        default:
          return;
      }
    }
  }
};


class ast_consumer : public clang::ASTConsumer{
  std::unique_ptr<general_decl_visitor> visitor;
public:
  explicit ast_consumer(clang::CompilerInstance& ci) : visitor{
    new general_decl_visitor{
      not_handled_ostream,
        ci.getASTContext().getPrintingPolicy(),
        ci.getASTContext()
    }
  }
  {
    ci.getPreprocessor().addPPCallbacks(std::make_unique<preprocessor_defines_extractor>(
            preprocessor_defines_ostream
          ));
  }
  virtual void HandleTranslationUnit(clang::ASTContext& context)override{
    visitor->Visit(context.getTranslationUnitDecl());
  }
};

struct ast_frontend_action : clang::SyntaxOnlyAction{
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& ci, clang::StringRef)override{
    return std::make_unique<ast_consumer>(ci);
  }
};

} // namespace headercvt

int main(int argc, const char** argv){
  llvm::cl::OptionCategory tool_category("headercvt options");
  llvm::cl::extrahelp common_help(clang::tooling::CommonOptionsParser::HelpMessage);
  std::vector<const char*> params;
  params.reserve(argc+1);
  std::copy(argv, argv+argc, std::back_inserter(params));
  params.emplace_back("-xc");
  params.emplace_back("-w");
  clang::tooling::CommonOptionsParser options_parser(argc = static_cast<int>(params.size()), params.data(), tool_category);
  clang::tooling::ClangTool tool(options_parser.getCompilations(), options_parser.getSourcePathList());

  {
    // setup files and their indentations
    using namespace headercvt;

    preprocessor_defines_ostream.without_using() << "cdef extern from \"CL/cl.h\":\n";
    preprocessor_defines_indentation ++;
    preprocessor_defines_ostream.without_using() << indent_str << "cdef enum:\n";
    preprocessor_defines_indentation ++;

    func_decl_ostream.without_using() << "cdef extern from \"CL/cl.h\":\n";
    func_decl_indentation ++;

    types_ostream.without_using() << "cdef extern from \"CL/cl.h\":\n";
    types_indentation ++;
  }

  auto const result_value = tool.run(clang::tooling::newFrontendActionFactory<headercvt::ast_frontend_action>().get());

  {
    using namespace headercvt;
    if (preprocessor_defines_ostream.vacant()){
      preprocessor_defines_ostream << indent_str << indent_str << "pass\n";
    }
    if (func_decl_ostream.vacant()){
      func_decl_ostream << indent_str << "pass\n";
    }
    if (types_ostream.vacant()){
      types_ostream << indent_str << "pass\n";
    }
  }

  headercvt::func_decl_ostream.flush();
  headercvt::types_ostream.flush();
  headercvt::not_handled_ostream.flush();
  headercvt::preprocessor_defines_ostream.flush();

  return result_value;
}
