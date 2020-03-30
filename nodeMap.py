
from collections import defaultdict
import _pickle as pkl 
node_list = ['unit_kind', 'decl', 'decl_stmt', 'init', 'expr', 'expr_stmt', 'comment', 'call', 'control', 'incr',
            'none', 'variable', 'function', 'function_decl', 'constructor', 'constructor_decl', 'destructor',
            'destructor_decl', 'macro', 'single_macro', 'nulloperator', 'enum_defn', 'enum_decl', 'global_attribute',
            'property_accessor', 'property_accessor_decl', 'expression', 'class_defn', 'class_decl', 'union_defn', 'union_decl',
            'struct_defn', 'struct_decl', 'interface_defn', 'interface_decl', 'access_region', 'using', 'operator_function',
            'operator_function_decl', 'event', 'property', 'annotation_defn', 'global_template', 'unit', 'tart_element_token', 'nop', 'string',
            'char', 'literal', 'boolean', 'null', 'complex', 'operator', 'modifier', 'name', 'oname', 'cname', 'type', 'typeprev', 'condition',
            'block', 'pseudo_block', 'index', 'enum', 'enum_declaration', 'if_statement', 'ternary', 'then', 'else', 'elseif', 'while_statement',
            'do_statement', 'for_statement', 'foreach_statement', 'for_control', 'for_initialization', 'for_condition', 'for_increment', 'for_like_control',
            'expression_statement', 'function_call', 'declaration_statement', 'declaration', 'declaration_initialization', 'declaration_range', 'range',
            'goto_statement', 'continue_statement', 'break_statement', 'label_statement', 'label', 'switch', 'case', 'default', 'function_definition',
            'function_declaration', 'lambda', 'function_lambda', 'function_specifier', 'return_statement', 'parameter_list', 'parameter', 'krparameter_list',
            'krparameter', 'argument_list', 'argument', 'pseudo_parameter_list', 'indexer_parameter_list', 'class', 'class_declaration', 'struct',
            'struct_declaration', 'union', 'union_declaration', 'derivation_list', 'public_access', 'public_access_default', 'private_access',
            'private_access_default', 'protected_access', 'protected_access_default', 'member_init_list', 'member_initialization_list', 'member_initialization',
            'constructor_definition', 'constructor_declaration', 'destructor_definition', 'destructor_declaration', 'friend', 'class_specifier', 'try_block',
            'catch_block', 'finally_block', 'throw_statement', 'throw_specifier', 'throw_specifier_java', 'template', 'generic_argument', 'generic_argument_list',
            'template_parameter', 'template_parameter_list', 'generic_parameter', 'generic_parameter_list', 'typedef', 'asm', 'macro_call', 'sizeof_call', 'extern',
            'namespace', 'using_directive', 'directive', 'atomic', 'static_assert_statement', 'generic_selection', 'generic_selector', 'generic_association_list',
            'generic_association', 'alignas', 'decltype', 'capture', 'lambda_capture', 'noexcept', 'typename', 'alignof', 'typeid', 'sizeof_pack', 'enum_class',
            'enum_class_declaration', 'ref_qualifier', 'signal_access', 'forever_statement', 'emit_statement', 'cpp_directive', 'cpp_filename', 'file', 'number',
            'cpp_number', 'cpp_literal', 'cpp_macro_defn', 'cpp_macro_value', 'error', 'cpp_error', 'cpp_warning', 'cpp_pragma', 'cpp_include', 'cpp_define',
            'cpp_undef', 'cpp_line', 'cpp_if', 'cpp_ifdef', 'cpp_ifndef', 'cpp_then', 'cpp_else', 'cpp_elif', 'cpp_empty', 'cpp_region', 'cpp_endregion',
            'using_stmt', 'escape', 'value', 'cpp_import', 'cpp_endif', 'marker', 'error_parse', 'error_mode', 'implements', 'extends', 'import', 'package',
            'assert_statement', 'interface', 'interface_declaration', 'synchronized_statement', 'annotation', 'static_block', 'checked_statement',
            'unchecked_statement', 'attribute', 'target', 'unsafe_statement', 'lock_statement', 'fixed_statement', 'typeof', 'using_statement',
            'function_delegate', 'constraint', 'linq', 'from', 'where', 'select', 'let', 'orderby', 'join', 'group', 'in', 'on', 'equals', 'by',
            'into', 'empty', 'empty_stmt', 'receiver', 'message', 'selector', 'protocol_list', 'category', 'protocol', 'required_default',
            'required', 'optional', 'attribute_list', 'synthesize', 'dynamic', 'encode', 'autoreleasepool', 'compatibility_alias', 'nil',
            'class_interface', 'class_implementation', 'protocol_declaration', 'cast', 'const_cast', 'dynamic_cast', 'reinterpret_cast',
            'static_cast', 'position', 'cuda_argument_list', 'omp_directive', 'omp_name', 'omp_clause', 'omp_argument_list', 'omp_argument',
            'omp_expression', 'end_element_token', 'main', 'break', 'continue', 'while', 'do', 'for', 'if', 'goto', 'visual_cxx_asm', 'sizeof',
            'auto', 'register', 'restrict', 'imaginary', 'noreturn', 'static_assert', 'crestrict', 'cxx_try', 'cxx_catch', 'cxx_class', 'constexpr',
            'thread_local', 'nullptr', 'void', 'return', 'include', 'define', 'elif', 'endif', 'errorprec', 'warning', 'ifdef', 'ifndef', 'line', 'pragma',
            'undef', 'inline', 'macro_type_name', 'macro_case', 'macro_label', 'specifier', 'try', 'catch', 'throw', 'throws', 'public', 'private',
            'protected', 'virtual', 'explicit', 'forever', 'signal', 'emit', 'new', 'delete', 'static', 'const', 'mutable', 'volatile', 'transient',
            'finally', 'final', 'abstract', 'super', 'synchronized', 'native', 'strictfp', 'nullliteral', 'assert', 'foreach', 'ref', 'out', 'lock', 'is', 'internal', 'sealed', 'override', 'implicit', 'stackalloc', 'as', 'delegate', 'fixed', 'checked', 'unchecked', 'region', 'endregion', 'unsafe', 'readonly', 'get', 'set', 'add', 'remove', 'yield', 'partial', 'await', 'async', 'this', 'params', 'alias', 'ascending', 'descending', 'atinterface', 'atimplementation', 'atend', 'atprotocol', 'atrequired', 'atoptional', 'atclass', 'weak',
            'strong', 'omp_omp', 'special_chars', 'slice_define', 'slice_use']


node_dict=defaultdict(int)
for node in node_list:
    node_dict[node]=len(node_dict)+1



