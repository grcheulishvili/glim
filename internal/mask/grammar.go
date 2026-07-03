package mask

import (
	"fmt"
	"sort"
	"strings"
)

// GenerateJSONSchemaGBNF takes simple key-type pairs and outputs a strict JSON GBNF string.
// Supported primitive types: "string", "number"
func GenerateJSONSchemaGBNF(schema map[string]string) string {
	var rules []string
	var rootElements []string

	// Sort keys to guarantee deterministic grammar ruleset layout
	keys := make([]string, 0, len(schema))
	for k := range schema {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		valType := schema[key]
		escapedKey := fmt.Sprintf(`"\\\"%s\\\""`, key)
		valueRuleName := fmt.Sprintf("%s-value", key)

		rootElements = append(rootElements, fmt.Sprintf(`%s ":" space %s`, escapedKey, valueRuleName))

		switch strings.ToLower(valType) {
		case "number":
			rules = append(rules, fmt.Sprintf(`%s ::= [0-9]+ ("." [0-9]+)?`, valueRuleName))
		case "string":
			fallthrough
		default:
			rules = append(rules, fmt.Sprintf(`%s ::= "\"" [^\"]* "\""`, valueRuleName))
		}
	}

	// Adjusted root: stripped opening "{\n" since it is injected directly into the context prompt matrix
	rootRule := fmt.Sprintf("root ::= space %s \"\n}\"", strings.Join(rootElements, ` ",\n" space `))

	baseRules := []string{
		rootRule,
		`space ::= " "*`,
	}
	baseRules = append(baseRules, rules...)

	return strings.Join(baseRules, "\n")
}
