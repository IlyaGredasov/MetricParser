import math
from collections import defaultdict
import pandas as pd

from pyparsing import Word, alphas, alphanums, oneOf, ZeroOrMore, ParserElement, OneOrMore, Group, Literal, \
    Optional, Forward, infixNotation, opAssoc, delimitedList, QuotedString, ParseResults

ParserElement.enablePackrat()

simple_operators = [
    '+', '-', '*', '/', '//', '%', '++', '--', '+=', '-=', '*=', '/=', '%=',
    '&', '|', '^', '~', '<<', '>>', '<<=', '>>=', '&=', '|=', '^=',
    '==', '!=', '>', '<', '>=', '<=', '&&', '||', '!', '->', ',', '='
]
type_words = [
    "int", "int64_t", "float", "double", "MatrixXd", "auto"
]
modifier_words = [
    "constexpr", "const", "volatile"
]
operator_expr = oneOf(simple_operators)
modifiers_words = oneOf(modifier_words)("modifiers")
modifiers_expr = Group(ZeroOrMore(modifiers_words))
simple_identifier_expr = Word(alphas + "_", alphanums + "_")
literal_const_expr = (QuotedString('"', escChar='\\', unquoteResults=False) |
                      QuotedString("'", escChar='\\', unquoteResults=False) |
                      Word("0123456789.-"))
include_identifier_expr = Word(alphas + "_", alphanums + "_./")("include_identifier")

composite_identifier_expr = Group(
    simple_identifier_expr + ZeroOrMore(oneOf(["::", "."]) + simple_identifier_expr))("identifier")

type_words_expr = oneOf(type_words) | composite_identifier_expr

template_type_expr = (
        composite_identifier_expr("template_name")
        + Literal("<")
        + Group(delimitedList(type_words_expr | literal_const_expr))("template_params")
        + Literal(">")
)("template_type")

ref_type_expr = (
        oneOf(["*", "**", "&", "&&"])
        + composite_identifier_expr("ref")
)("ref_type")

lvalue_expr = Forward()("lvalue")
rvalue_expr = Forward()("rvalue")
function_call_expr = Forward()("function_call")

initializer_list_expr = Group(
    Literal("{") + delimitedList(rvalue_expr) + Literal("}")
)("initializer_list")

static_cast_expr = Group(
    Literal("static_cast")
    + Literal("<")
    + type_words_expr("cast_type")
    + Literal(">")
    + Literal("(")
    + rvalue_expr("value")
    + Literal(")")
)("static_cast")

paren_expr = (Literal("(") + rvalue_expr + Literal(")"))("paren")

rvalue_term = Group(
    paren_expr |
    static_cast_expr |
    lvalue_expr |
    literal_const_expr |
    initializer_list_expr |
    function_call_expr |
    simple_identifier_expr
)("rvalue")

rvalue_expr <<= infixNotation(
    rvalue_term,
    [
        (oneOf('- ! ~'), 1, opAssoc.RIGHT, lambda tokens: [tokens[0][0], tokens[0][1]]),
        (oneOf('* / % + - < <= > >= == != && || << >>'), 2, opAssoc.LEFT),
        (oneOf('= += -= *= /= %= <<= >>= &= |= ^= ,'), 2, opAssoc.RIGHT),
    ]
)("rvalue")

function_call_expr <<= Group(
    composite_identifier_expr("func_name")
    + Literal("(")
    + Optional(delimitedList(rvalue_expr))("args")
    + Literal(")")
)("func_call")

chained_method_expr = (
        (function_call_expr | composite_identifier_expr)("base")
        + ZeroOrMore(Literal(".") + (function_call_expr | composite_identifier_expr))
)("chained_method")

array_subscription_expr = (
        simple_identifier_expr("array_name")
        + Literal("[")
        + rvalue_expr("index_expr")
        + Literal("]")
)("array_subscription")

lvalue_expr <<= (
        array_subscription_expr |
        chained_method_expr |
        function_call_expr |
        composite_identifier_expr |
        template_type_expr |
        simple_identifier_expr |
        ref_type_expr
)

lvalue_action_statement_expr = Group(
    Optional(ZeroOrMore(modifiers_words))
    + Optional(type_words_expr("type"))
    + lvalue_expr("var")
    + operator_expr("op")
    + rvalue_expr("rvalue")
    + Literal(";")
)("lvalue_action_statement")

rvalue_action_statement_expr = Group(
    rvalue_expr("expr")
    + Literal(";")
)("rvalue_action_statement")

statement_expr = Forward()("statement")

include_expr = Group(
    Literal("#include")
    + Literal("<")
    + include_identifier_expr
    + Literal(">")
)("include")

using_expr = Group(
    Literal("using")
    + lvalue_expr("alias")
    + Optional(Literal("=") + template_type_expr("template_type"))
    + Literal(";")
)("using")

if_expr = Forward()("if_block")
for_cycle_expr = Forward()("for_cycle")

return_expr = Group(
    Literal("return")
    + rvalue_expr("value")
    + Literal(";")
)("return_statement")

if_expr <<= Group(
    Literal("if")
    + Literal("(")
    + rvalue_expr("if_condition")
    + Literal(")")
    + (
            Group(Literal("{") + ZeroOrMore(statement_expr)("then_body") + Literal("}"))("then_block")
            | statement_expr("then_body")
    )
    + Optional(
        Literal("else")
        + (
                Group(Literal("{") + ZeroOrMore(statement_expr)("else_body") + Literal("}"))("else_block")
                | statement_expr("else_body")
        )
    )
)

for_init_expr = Group(
    Optional(type_words_expr)("type")
    + Group(simple_identifier_expr("var"))("identifier")
    + Literal("=")
    + literal_const_expr("start")
)("for_init")

for_condition_expr = Group(
    Group(simple_identifier_expr("cond_var"))("identifier")
    + oneOf("< <= > >=")("operator")
    + Group(simple_identifier_expr("cond_val"))("identifier")
)("for_condition")

for_update_expr = Group(
    Optional("++")
    + lvalue_expr("upd_var")
    + Optional(oneOf("++ -- += -="))("update")
)("for_update")

for_cycle_expr <<= Group(
    Literal("for")
    + Literal("(")
    + for_init_expr("for_init")
    + Literal(";")
    + for_condition_expr("for_condition")
    + Literal(";")
    + for_update_expr("for_update")
    + Literal(")")
    + Literal("{")
    + ZeroOrMore(statement_expr)("body")
    + Literal("}")
)("for_cycle")

statement_expr <<= (
        lvalue_action_statement_expr | rvalue_action_statement_expr | if_expr | for_cycle_expr | return_expr)

func_arg_expr = Group(
    Optional(Group(ZeroOrMore(modifiers_words)))
    + type_words_expr("type")
    + Group(simple_identifier_expr | ref_type_expr)("identifier")
)("func_arg")

func_definition_expr = Group(
    Optional(Group(ZeroOrMore(modifiers_words)))
    + composite_identifier_expr("modifiers")
    + Group(simple_identifier_expr)("func_name")
    + Literal("(")
    + Group(Optional(delimitedList(func_arg_expr)))("args_list")
    + Literal(")")
    + Literal("{")
    + ZeroOrMore(statement_expr | for_cycle_expr)("body")
    + Optional(return_expr)("return_type")
    + Literal("}")
)("func_definition")

token_expr = (
        include_expr.setResultsName("include", listAllMatches=True) |
        using_expr.setResultsName("using", listAllMatches=True) |
        statement_expr.setResultsName("statement", listAllMatches=True) |
        for_cycle_expr.setResultsName("for_loop", listAllMatches=True) |
        func_definition_expr.setResultsName("func_definition", listAllMatches=True) |
        if_expr.setResultsName("if", listAllMatches=True)
)
cpp_grammar = OneOrMore(token_expr)
cpp_grammar = cpp_grammar("root")


def parse(cpp_code: str) -> None:
    distinct_operators = defaultdict(int)
    distinct_operands = defaultdict(int)

    def add_operators(parsed):
        for token in parsed:
            if token in simple_operators:
                distinct_operators[token] += 1

    def parsed_to_string(parsed):
        return "".join([str(token) for token in parsed])

    def add_operands(parsed):
        for token in parsed:
            if isinstance(token, str):
                distinct_operands[token] += 1

    def walk_parse_results(parsed):
        if isinstance(parsed, ParseResults):
            name = parsed.getName()
            if name != "root":
                print(name, parsed)

            match name:
                case "for_condition":
                    add_operators(parsed)

                case "for_init":
                    add_operators(parsed)

                case "for_update":
                    add_operators(parsed)

                case "func_call":
                    distinct_operators["()"] += 1

                case "func_definition":
                    distinct_operands[parsed_to_string(parsed.get("func_name"))] += 1

                case "func_name":
                    add_operators(parsed)
                    if "." not in parsed and parsed in distinct_operands:
                        add_operands(parsed)

                case "identifier":
                    add_operands([parsed_to_string(parsed)])

                case "initializer_list":
                    add_operators(parsed)

                case "lvalue_action_statement":
                    add_operators(parsed)

                case "rvalue":
                    add_operators(parsed)

                case "using":
                    if "=" in parsed_to_string(parsed):
                        distinct_operators["="] += 1

                case "template_params":
                    if len(parsed) > 1:
                        distinct_operators[","] += (len(parsed) - 1)

                case "args_list":
                    if len(parsed) > 1:
                        distinct_operators[","] += (len(parsed) - 1)
                case None:
                    distinct_operators["-"] += parsed_to_string(parsed).count("-")

            for sub in parsed:
                walk_parse_results(sub)

    parsed_tokens = cpp_grammar.parseString(cpp_code, parseAll=True)
    walk_parse_results(parsed_tokens)
    print(distinct_operators)
    print(distinct_operands)
    eta_1 = len(distinct_operators)
    eta_2 = len(distinct_operands)
    n_1 = sum(distinct_operators.values())
    n_2 = sum(distinct_operands.values())
    v = (n_1 + n_2) * math.log2(eta_1 + eta_2)
    df_operators = pd.DataFrame.from_dict(distinct_operators, orient="index")
    df_operands = pd.DataFrame.from_dict(distinct_operands, orient="index")
    df_operators.to_excel("Операторы.xlsx")
    df_operands.to_excel("Операнды.xlsx")


    print(f"""
    eta_1 (различных операторов): {eta_1},
    eta_2 (различных операндов): {eta_2},
    n_1 (всего операторов): {n_1},
    n_2 (всего операндов): {n_2},
    v (volume)': {v}
    """
          )


if __name__ == "__main__":
    with open("D:/Programming/C++/EigenMetrology/main.cpp", encoding="utf-8") as file:
        source = "".join(file.readlines())
        parse(source)
