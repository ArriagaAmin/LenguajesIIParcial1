from typing import *

class OperatorGrammar:
  """
  Implementacion de una Gramatica de Operadores.
  """

  def __init__(self):
    # Gramatica G = (N, Sigma, S, P)
    self.N = set()
    self.Sigma = set(["$"])
    self.S = None
    self.P = {}
    self.P_ref = {}
    # Tabla y funciones para las precedencias
    self.precedences = {}
    self.equiv_graph = {}
    self.equivs = {}
    self.equivs_reverse = {}
    self.nodes = {}
    self.f = {}
    self.g = {}
    # Verificar si se realizo build
    self.builded = False

  def is_terminal(self, symbol: str, accept_dolar: bool = False) -> bool:
    """
      Verifica si un simbolo es terminal. Un simbolo se considera terminal cuando
      esta en minuscula o es un signo en ASCII, y no-terminal si es un simbolo en
      mayusculas. En caso contrario, el simbolo es invalido.

      INPUTS:
        symbol: str         ->  Simbolo a analizar.
        accept_dolar: bool  ->  Indica si se puede analizar simbolo $. Valor por 
            defecto: False.
      OUTPUT:
        bool  ->  True si es terminal, False si es no-terminal

      >>> OG = OperatorGrammar()
      >>> OG.is_terminal("$")
      Traceback (most recent call last):
      ...
      Exception: El simbolo \033[1;3m$\033[0m esta reservado y no puede ser usado en ninguna regla.
      >>> OG.is_terminal("$", True)
      True
      >>> OG.is_terminal("z")
      True
      >>> OG.is_terminal("`")
      True
      >>> OG.is_terminal("\\"")
      True
      >>> OG.is_terminal("K")
      False
      >>> OG.is_terminal("á")
      Traceback (most recent call last):
      ...
      Exception: El simbolo \033[1;3má\033[0m no es valido. 
      Utilice caracteres y signos de la tabla ASCII, excepto el \033[1;3m$\033[0m.
      >>> OG.is_terminal("ay no")
      Traceback (most recent call last):
      ...
      Exception: El simbolo \033[1;3may no\033[0m no es valido.
      Los simbolos deben estar compuestos de un solo caracter
    """

    if len(symbol) > 1:
      raise Exception(
        f'El simbolo \033[1;3m{symbol}\033[0m no es valido.\nLos simbolos deben ' + \
        'estar compuestos de un solo caracter'
      )

    ascii_value = ord(symbol)

    # Verificamos que no sea el simbolo $
    if ascii_value == 36 and not accept_dolar:
      raise Exception(
        f'El simbolo \033[1;3m$\033[0m esta reservado y no puede ser usado en ' + \
        'ninguna regla.'
      )

    # Los simbolos y caracteres en minusculas en ASCII estan en el
    # rango [33, 65) U [91, 127)
    if (33 <= ascii_value < 65) or (91 <= ascii_value < 127): return True
    # Mientras que los caracteres en mayusculas estan en el rango [65, 91)
    elif (65 <= ascii_value < 91): return False 
    # Si no esta en ninguno de esos rangos, no es un simbolo aceptado.
    raise Exception(
      f'El simbolo \033[1;3m{symbol}\033[0m no es valido. \n' + \
      'Utilice caracteres y signos de la tabla ASCII, excepto el \033[1;3m$\033[0m.'
    )

  def set_init(self, S: str):
    """
      Asigna a un simbolo no-terminal como simbolo inicial de la gramatica.

      INPUTS:
        S: str  ->  Simbolo a establecer como inicial.
      
      >>> OG = OperatorGrammar()
      >>> OG.set_init("a")
      Traceback (most recent call last):
      ...
      Exception: El simbolo \033[1;3ma\033[0m es terminal. 
      El simbolo inicial debe ser no-terminal.
      >>> OG.set_init("A")
      >>> OG.N
      {'A'}
      >>> OG.set_init("S")
      Traceback (most recent call last):
        File "/usr/lib/python3.7/doctest.py", line 1329, in __run
          compileflags, 1), test.globs)
        File "<doctest __main__.OperatorGrammar.set_init[3]>", line 1, in <module>
          OG.set_init("S")
        File "OperatorGrammarAnalyzer.py", line 115, in set_init
          'No pueden haber 2 o mas simbolos inciales.'
      Exception: Ya se definio al simbolo \033[1;3mA\033[0m como inicial. 
      No pueden haber 2 o mas simbolos inciales.
    """

    # Verificamos que el simbolo es no-terminal
    if self.is_terminal(S):
      raise Exception(
        f'El simbolo \033[1;3m{S}\033[0m es terminal. \nEl simbolo inicial debe ' + \
        'ser no-terminal.'
      )

    # No se debe haber definido otro simbolo inicial anteriormente.
    if self.S != None:
      raise Exception(
        f'Ya se definio al simbolo \033[1;3m{self.S}\033[0m como inicial. \n' + \
        'No pueden haber 2 o mas simbolos inciales.'
      )

    # Lo agregamos a los simbolos no terminales
    self.N.add(S)

    self.S = S

  def verify_operator_grammar_rule(self, rule: List[str]):
    """
      Verifica que una produccion cumpla la regla de no tener 2 simbolos no-terminales
      seguidos, y almacena los simbolos que conforman la produccion en sus
      correspondientes conjuntos.

      INPUTS:
        rule: List[str] ->  Lista de tokens que conforman la regla.

      >>> OG = OperatorGrammar()
      >>> OG.verify_operator_grammar_rule(["a", "A", "b", "B"])
      >>> OG.verify_operator_grammar_rule(["~", "Z", "K"])
      Traceback (most recent call last):
      ...
      Exception: La regla contiene dos simbolos no-terminales \033[1;3mZ\033[0m y \033[1;3mK\033[0m seguidos. 
      Recuerde que una \033[3mGramatica de Operadores\033[0m no debe contener dos simbolos no-terminales seguidos.
      >>> OG.verify_operator_grammar_rule(["R", "2", "D", "2", "~"])
      >>> N = list(OG.N)
      >>> N.sort()
      >>> N
      ['A', 'B', 'D', 'R']
      >>> Sigma = list(OG.Sigma)
      >>> Sigma.sort()
      >>> Sigma
      ['$', '2', 'a', 'b', '~']
    """

    last_is_terminal = current_is_terminal = True

    Sigma = set()
    N = set()

    for i, symbol in enumerate(rule):
      current_is_terminal = self.is_terminal(symbol)
      # Verificamos que alguno entre el simbolo anterior y el actual sea terminal
      if not (last_is_terminal or current_is_terminal):
        raise Exception(
          f'La regla contiene dos simbolos no-terminales \033[1;3m{rule[i-1]}\033[0m ' + \
          f'y \033[1;3m{symbol}\033[0m seguidos. \nRecuerde que una \033[3mGramatica de ' + \
          'Operadores\033[0m no debe contener dos simbolos no-terminales seguidos.'
        )

      # Agregamos el simbolo al conjunto correspondiente
      if current_is_terminal: Sigma.add(symbol)
      else: N.add(symbol)

      last_is_terminal = current_is_terminal

    for s in Sigma: self.Sigma.add(s)
    for n in N: self.N.add(n)

  def get_rule(self, right_rule: Tuple[str]) -> str:
    """
      Obtenemos la representacion de una regla dada la tupla con los tokens
      que representan el lado derecho de esta.

      INPUTS:
        right_rule: Tuple[str]  ->  Tupla con los tokens del lado derecho de la regla
      OUTPUT:
        str ->  Representacion de la regla.

      >>> OG = OperatorGrammar()
      >>> OG.get_rule(("E", "+", "E"))
      Traceback (most recent call last):
      ...
      Exception: No existe una regla cuyo lado derecho sea \033[1;3mE + E \033[0m.
      >>> OG.P_ref[("E", "+", "E")] = "S"
      >>> OG.get_rule(("E", "+", "E"))
      'S -> E + E '
    """

    right = ""
    for token in right_rule: right += token + " "
    
    if not right_rule in self.P_ref:
      raise Exception(
        f'No existe una regla cuyo lado derecho sea \033[1;3m{right}\033[0m.'
      )

    return self.P_ref[right_rule] + " -> " + right

  def make_rule(self, rule: str):
    """
      Agrega una produccion a la gramatica.

      INPUTS:
        rule: str ->  String que representa la regla.

      >>> OG = OperatorGrammar()
      >>> OG.make_rule("E E + E * 1")
      'E -> E + E * 1 '
      >>> OG.make_rule("a 1")
      Traceback (most recent call last):
      ...
      Exception: El simbolo \033[1;3ma\033[0m es terminal. 
      El lado izquierdo de toda regla debe ser un simbolo no-terminal.
      
      >>> OG.make_rule("S")
      \033[1;35mWarning:\033[0m Esta regla es una lambda-produccion. 
      Se considerara al simbolo \033[1;3mS\033[0m como inicial para cumplir la definicion de \033[3mGramatica de Operadores\033[0m.
      'S -> '

      >>> OG.make_rule("E")
      Traceback (most recent call last):
      ...
      Exception: Esta regla es una lambda-produccion y la gramatica ya tiene un simbolo inicial. 
      Recuerde que una \033[3mGramatica de Operadores\033[0m no debe contener lambda-producciones salvo quiza para el simbolo inicial.
      
      >>> N = list(OG.N)
      >>> N.sort()
      >>> N
      ['E', 'S']
      
      >>> Sigma = list(OG.Sigma)
      >>> Sigma.sort()
      >>> Sigma
      ['$', '*', '+', '1']

      >>> P = list(OG.P)
      >>> P.sort()
      >>> [(p, OG.P[p]) for p in P]
      [('E', {('E', '+', 'E', '*', '1')}), ('S', {()})]

      >>> P_ref = list(OG.P_ref)
      >>> P_ref.sort()
      >>> [(p, OG.P_ref[p]) for p in P_ref]
      [((), 'S'), (('E', '+', 'E', '*', '1'), 'E')]
    """

    tokens = rule.split()
    no_terminal = tokens.pop(0)

    # Verificamos que el lado izquierdo de la regla es no-terminal.
    if self.is_terminal(no_terminal):
      raise Exception(
        f'El simbolo \033[1;3m{no_terminal}\033[0m es terminal. \nEl lado ' + \
        'izquierdo de toda regla debe ser un simbolo no-terminal.'
      )

    # Verificamos si la regla es una lambda-produccion
    if (len(tokens) == 0) and (self.S != None):
      raise Exception(
        f'Esta regla es una lambda-produccion y la gramatica ya tiene un simbolo ' + \
        'inicial. \nRecuerde que una \033[3mGramatica de Operadores\033[0m no debe ' + \
        'contener lambda-producciones salvo quiza para el simbolo inicial.'
      )
    elif (len(tokens) == 0):
      print(
        '\033[1;35mWarning:\033[0m Esta regla es una lambda-produccion. \nSe considerara ' + \
        f'al simbolo \033[1;3m{no_terminal}\033[0m como inicial para cumplir la ' + \
        'definicion de \033[3mGramatica de Operadores\033[0m.'
      )
      self.set_init(no_terminal)

    self.verify_operator_grammar_rule(tokens)

    # Agregamos la referencia  rule -> no-terminal
    self.P_ref[tuple(tokens)] = no_terminal
    self.N.add(no_terminal)

    if no_terminal in self.P: self.P[no_terminal].add(tuple(tokens))
    else: self.P[no_terminal] = set([tuple(tokens)])

    return self.get_rule(tuple(tokens))

  def set_precedence(self, left: str, op: str, right: str):
    """
      Establece una relacion de precedencia entre dos simbolos terminales.

      INPUTS:
        left: str   ->  Lado izquierdo de la relacion.
        op: str     ->  Relacion
        right: str  ->  Lado derecho de la relacion.

      >>> OG = OperatorGrammar()
      >>> OG.set_precedence("P", ">", "+")
      Traceback (most recent call last):
      ...
      Exception: El simbolo \033[1;3mP\033[0m no es terminal. 
      Solo se puede establecer precedencia entre simbolos terminales.

      >>> OG.set_precedence("+", ">", "S") 
      Traceback (most recent call last):
      ...
      Exception: El simbolo \033[1;3mS\033[0m no es terminal. 
      Solo se puede establecer precedencia entre simbolos terminales.

      >>> OG.set_precedence("+", ">=", "*") 
      Traceback (most recent call last):
      ...
      Exception: Operador \033[1;3m>=\033[0m invalido. 
      Los unicos operadores disponibles para establecer precedencias son \033[1;3m<\033[0m, \033[1;3m=\033[0m y \033[1;3m>\033[0m.

      >>> OG.set_precedence("+", ">", "*") 
      >>> OG.set_precedence("*", "<", "+") 
      >>> OG.set_precedence("+", "=", "*") 
      \033[1;35mWarning:\033[0m Ya se definio la relacion de precedencia \033[1;3m>\033[0m entre \033[1;3m+\033[0m y \033[1;3m*\033[0m.

      >>> OG.set_precedence("+", "=", "+")
      >>> OG.set_precedence("+", ">", "+")
      \033[1;35mWarning:\033[0m Ya se definio la relacion de precedencia \033[1;3m=\033[0m entre \033[1;3m+\033[0m y \033[1;3m+\033[0m.
    """

    # Verificamos que los simbolos sean terminales
    if not self.is_terminal(left, True):
      raise Exception(
        f'El simbolo \033[1;3m{left}\033[0m no es terminal. \nSolo se puede establecer ' + \
        'precedencia entre simbolos terminales.'
      )
    if not self.is_terminal(right, True):
      raise Exception(
        f'El simbolo \033[1;3m{right}\033[0m no es terminal. \nSolo se puede establecer ' + \
        'precedencia entre simbolos terminales.'
      )

    # Verificamos que el operador sea valida
    if not op in {'<', '=', '>'}:
      raise Exception(
        f'Operador \033[1;3m{op}\033[0m invalido. \nLos unicos operadores disponibles para ' + \
        'establecer precedencias son \033[1;3m<\033[0m, \033[1;3m=\033[0m y ' + \
        '\033[1;3m>\033[0m.'
      )

    # Verificamos que no se haya establecido otra precedencia
    if left in self.precedences:
      if right in self.precedences[left]:
        print(
          '\033[1;35mWarning:\033[0m Ya se definio la relacion de precedencia ' + \
          f'\033[1;3m{self.precedences[left][right]}\033[0m entre \033[1;3m{left}\033[0m ' + \
          f'y \033[1;3m{right}\033[0m.'
        )
        self.precedences[left].pop(right)
    if (left, True) in self.equiv_graph:
      if (right, False) in self.equiv_graph[(left, True)]:
        print(
          '\033[1;35mWarning:\033[0m Ya se definio la relacion de precedencia ' + \
          f'\033[1;3m=\033[0m entre \033[1;3m{left}\033[0m y \033[1;3m{right}\033[0m.'
        )
        self.equiv_graph[(left, True)].discard((right, False))
        self.equiv_graph[(right, False)].discard((left, True))

  
    # Si el operador es =, agregamos a right y left a la misma clase de equivalencia.
    if op == '=':

      # Agregamos la relacion de equivalencia f_left -> g_right
      if (left, True) in self.equiv_graph: self.equiv_graph[(left, True)].add((right, False))
      else: self.equiv_graph[(left, True)] = set([(right, False)])
      # Agregamos la relacion de equivalencia g_right -> f_left
      if (right, False) in self.equiv_graph: self.equiv_graph[(right, False)].add((left, True))
      else: self.equiv_graph[(right, False)] = set([(left, True)])

    else: 

      # Establecemos la precedencia
      if left in self.precedences: self.precedences[left][right] = op
      else: self.precedences[left] = {right: op}

    self.Sigma.add(left)
    self.Sigma.add(right)

  def dfs(self, symbol: Tuple[str, bool], visited: Set[Tuple[str, bool]], count: int) -> Set[str]:
    """
      Aplicamos DFS para crear la clases de equivalencia del simbolo.

      INPUTS:
        symbol: Tuple[str, bool]        ->  Simbolo (nodo) actual.
        visited: Set[Tuple[str, bool]]  ->  Nodos ya visitados.
        count: int                      ->  Clase de equivalencia actual.
      OUTPUT:
        Set[str]  ->  Nodos visitados luego de esta iteracion.

      >>> OG = OperatorGrammar()
      >>> OG.set_precedence("a", "=", "b")
      >>> OG.set_precedence("c", "=", "b")
      >>> OG.set_precedence("c", "=", "d")
      >>> OG.set_precedence("d", "=", "d")
      >>> OG.set_precedence("d", "=", "a")

      >>> s = list(OG.dfs(("a", True), set(), 0))
      >>> s.sort()
      >>> s
      [('a', False), ('a', True), ('b', False), ('c', True), ('d', False), ('d', True)]

      >>> [(t, OG.equivs[t]) for t in s]
      [(('a', False), 0), (('a', True), 0), (('b', False), 0), (('c', True), 0), (('d', False), 0), (('d', True), 0)]
    """

    self.equivs[symbol] = count
    if count in self.equivs_reverse: self.equivs_reverse[count].add(symbol)
    else: self.equivs_reverse[count] = set([symbol])

    visited.add(symbol) 

    # Si el simbolo no tiene ninguna relacion de equivalencia, entonces el es
    # su unico sub-grafo de equivalencia.
    if not symbol in self.equiv_graph: return visited

    for succ in self.equiv_graph[symbol]:
      if not succ in visited:
        visited = self.dfs(succ, visited, count)

    return visited
    
  def make_equiv_graph(self):
    """
      Construye las clases de equivalencia de la gramatica.

      >>> OG = OperatorGrammar()
      >>> OG.set_precedence("a", "=", "b")
      >>> OG.set_precedence("c", "=", "b")
      >>> OG.set_precedence("e", "=", "d")
      >>> OG.set_precedence("d", "=", "d")
      >>> OG.set_precedence("f", "=", "a")
      >>> OG.make_equiv_graph()

      >>> s = list(OG.equivs)
      >>> s.sort()

      >>> for e in s:
      ...   if OG.equivs[e] == OG.equivs[('e', True)]: print(e)
      ('d', False)
      ('d', True)
      ('e', True)

      >>> for e in s:
      ...   if OG.equivs[e] == OG.equivs[('b', False)]: print(e)
      ('a', True)
      ('b', False)
      ('c', True)

      >>> for e in s:
      ...   if OG.equivs[e] == OG.equivs[('a', False)]: print(e)
      ('a', False)
      ('f', True)
    """

    visited = set()
    count = 0

    # Aplicamos DFS para enumerar las clases de equivalencia
    for symbol in self.Sigma:
      if not (symbol, True) in visited:
        visited = self.dfs((symbol, True), visited, count)
        count += 1
      if not (symbol, False) in visited:
        visited = self.dfs((symbol, False), visited, count)
        count += 1

  def make_precedence_graph(self):
    """
      Construye el grafo de precedencia entre las clases de equivalencias.

      >>> OG = OperatorGrammar()
      >>> OG.set_precedence("a", "=", "b")
      >>> OG.set_precedence("b", "=", "c")
      >>> OG.set_precedence("f", "=", "e")
      >>> OG.set_precedence("a", "<", "z")
      >>> OG.set_precedence("f", ">", "z")
      >>> OG.set_precedence("f", "<", "a")
      >>> OG.set_precedence("f", ">", "q")
      >>> OG.make_precedence_graph()

      >>> OG.nodes[OG.equivs[('f', True)]] == set([OG.equivs[('z', False)], OG.equivs[('q', False)]])
      True
    """

    # Creamos las clases de equivalencia
    self.make_equiv_graph()

    # Creamos las relaciones de las clases de equivalencia
    for left in self.Sigma:
      if not left in self.precedences: continue

      for right in self.precedences[left]:
        
        # Si left < right, entonces  [g_right] -> [f_left] 
        if self.precedences[left][right] == '<':
          if self.equivs[(right, False)] in self.nodes: 
            self.nodes[self.equivs[(right, False)]].add(self.equivs[(left, True)])
          else: 
            self.nodes[self.equivs[(right, False)]] = set([self.equivs[(left, True)]])

        # En caso contrario (si es >), entonces  [f_left] -> [g_right]
        else:
          if self.equivs[(left, True)] in self.nodes: 
            self.nodes[self.equivs[(left, True)]].add(self.equivs[(right, False)])
          else: 
            self.nodes[self.equivs[(left, True)]] = set([self.equivs[(right, False)]])

  def get_node_str(self, node: int) -> str:
    """
      Retorna la representacion de una clase de equivalencia.
    """
    output = "{"
    for symbol in self.equivs_reverse[node]:
      output += "F"*symbol[1] + "G"*(not symbol[1]) + "_" + symbol[0] + ", "
    return output[:-2] + "}"

  def precedences_error(self, path: List[Tuple[int, bool]]):
    """
      Obtenemos el ciclo de precedencia y lo reportamos.
    """
    for i in range(len(path)-2, -1, -1): 
      if path[i] == path[-1]:
        index = i
        break 

    error = "Ha ocurrido un ciclo en las precedencias.\n" + \
      "A continuacion reportaremos las clases de equivalencias relacionadas:\n\n"

    # Creamos un string que representa el ciclo
    for i in range(index, len(path)-1):
      error += str(self.get_node_str(path[i])) + " -> "
    error += str(self.get_node_str(path[-1]))

    # Liberamos la memoria de los grafos de precedencia
    self.equiv_graph.clear()
    self.equivs.clear()
    self.equivs_reverse.clear()
    self.nodes.clear()

    # Reportamos el error
    raise Exception(error)

  def get_max_path(self, path: List[int], path_set: Set[int]) -> int:
    """
      Obtiene el maximo camino desde un simbolo (nodo) dado el grafo de equivalencias.

      >>> OG = OperatorGrammar()

      >>> OG.set_precedence("i", ">", "$")
      >>> OG.set_precedence("i", ">", "+")
      >>> OG.set_precedence("i", ">", "*")

      >>> OG.set_precedence("+", "<", "i")
      >>> OG.set_precedence("+", ">", "+")
      >>> OG.set_precedence("+", "<", "*")
      >>> OG.set_precedence("+", ">", "$")

      >>> OG.set_precedence("*", "<", "i")
      >>> OG.set_precedence("*", ">", "+")
      >>> OG.set_precedence("*", ">", "*")
      >>> OG.set_precedence("*", ">", "$")

      >>> OG.set_precedence("$", "<", "i")
      >>> OG.set_precedence("$", "<", "+")
      >>> OG.set_precedence("$", "<", "*")

      >>> OG.make_precedence_graph()
      >>> OG.get_max_path([OG.equivs[("i", False)]], set([OG.equivs[("i", False)]]))
      5
    """
 
    node = path[-1]

    symbol = list(self.equivs_reverse[node])[0]

    # Si ya calculamos el maximo camino, lo retornamos.
    if symbol[1] and symbol[0] in self.f: return self.f[symbol[0]] 
    if not symbol[1] and symbol[0] in self.g: return self.g[symbol[0]]

    # Obtenemos el maximo valor de los hijos.
    max_value = 0

    # Si no tiene nodos hijos, entonces su valor es 0
    if not node in self.nodes:
      for s in self.equivs_reverse[node]:
        if s[1]: self.f[s[0]] = 0
        else: self.g[s[0]] = 0
      return 0 

    # Por cada hijo del nodo actual.
    for child in self.nodes[node]:

      # Si el hijo ya esta en el camino actual, entonces hay un ciclo
      if child in path_set: 
        self.precedences_error(path + [child])

      path.append(child)
      path_set.add(child)
      max_value = max(max_value, self.get_max_path(path, path_set))
      path_set.discard(child)
      path.pop()

    for s in self.equivs_reverse[node]:
      if s[1]: self.f[s[0]] = max_value + 1
      else: self.g[s[0]] = max_value + 1
    return max_value + 1

  def make_precedence_functions(self):
    """
      Construye las funciones f y g de precedencia.

      >>> OG = OperatorGrammar()

      >>> OG.set_precedence("i", ">", "$")
      >>> OG.set_precedence("i", ">", "+")
      >>> OG.set_precedence("i", ">", "*")

      >>> OG.set_precedence("+", "<", "i")
      >>> OG.set_precedence("+", ">", "+")
      >>> OG.set_precedence("+", "<", "*")
      >>> OG.set_precedence("+", ">", "$")

      >>> OG.set_precedence("*", "<", "i")
      >>> OG.set_precedence("*", ">", "+")
      >>> OG.set_precedence("*", ">", "*")
      >>> OG.set_precedence("*", ">", "$")

      >>> OG.set_precedence("$", "<", "i")
      >>> OG.set_precedence("$", "<", "+")
      >>> OG.set_precedence("$", "<", "*")

      >>> OG.make_precedence_functions()

      >>> for s in ["$", "+", "*", "i"]:
      ...   print("f(" + s + ") = ", OG.f[s])
      f($) =  0
      f(+) =  2
      f(*) =  4
      f(i) =  4
      >>> for s in ["$", "+", "*", "i"]:
      ...   print("g(" + s + ") = ", OG.g[s])
      g($) =  0
      g(+) =  1
      g(*) =  3
      g(i) =  5

      >>> OG.set_precedence("a", "=", "b")
      >>> OG.set_precedence("b", "=", "a")
      >>> OG.set_precedence("a", "=", "a")
      >>> OG.set_precedence("d", "<", "c")
      >>> OG.set_precedence("d", ">", "e")
      >>> OG.set_precedence("a", "<", "e")
      >>> OG.set_precedence("b", ">", "c")
      >>> try:
      ...   OG.make_precedence_functions()
      ... except:
      ...   print("Hay un ciclo")
      Hay un ciclo
    """

    # Creamos el grafo de precedencia
    self.make_precedence_graph()

    # Limpiamos f y g por si guardan valores anteriores
    self.f = {}
    self.g = {}

    for s in self.Sigma:
      path = [self.equivs[(s, True)]]
      path_set = set([self.equivs[(s, True)]])
      self.get_max_path(path, path_set)

      path = [self.equivs[(s, False)]]
      path_set = set([self.equivs[(s, False)]])
      self.get_max_path(path, path_set)

    self.builded = True

    # Liberamos la memoria de los grafos de precedencia
    self.equiv_graph.clear()
    self.equivs.clear()
    self.equivs_reverse.clear()
    self.nodes.clear()

  def get_last_terminal(self, stack: List[str], upper: int) -> int:
    """
      Dada una lista de simbolos, retorna el indice del que se encuentre mas cerca y por
      debajo de una cota y sea terminal

      INPUTS:
        stack: List[str]  ->  Lista de simbolos.
        upper: int        ->  Cota superior.
      OUTPUT:
        int ->  Indice encontrado. -1 en caso de no encontrar ninguno.
    
      >>> OG = OperatorGrammar()
      >>> OG.get_last_terminal(["S", "S", "a", "S"], 3)
      2
      >>> OG.get_last_terminal(["S", "S", "a", "S"], 100)
      2
      >>> OG.get_last_terminal(["$", "S", "a", "S"], 1)
      0
      >>> OG.get_last_terminal(["S", "S", "a", "S"], 1)
      -1
    """

    upper = min(len(stack), upper)
    for i in range(upper-1, -1, -1):
      if self.is_terminal(stack[i], True): return i
    return -1

  def get_entry(self, w: str, index: int) -> str:
    """
      Retorna una entrada colocandole las relaciones de precedencia correspondientes.

      INPUTS:
        w: str  ->  Entrada original.
      OUTPUT:
        str -> Entrada con las relaciones de precedencia.

      >>> OG = OperatorGrammar()

      >>> OG.set_precedence("i", ">", "$")
      >>> OG.set_precedence("i", ">", "+")
      >>> OG.set_precedence("i", ">", "*")

      >>> OG.set_precedence("+", "<", "i")
      >>> OG.set_precedence("+", ">", "+")
      >>> OG.set_precedence("+", "<", "*")
      >>> OG.set_precedence("+", ">", "$")

      >>> OG.set_precedence("*", "<", "i")
      >>> OG.set_precedence("*", ">", "+")
      >>> OG.set_precedence("*", ">", "*")
      >>> OG.set_precedence("*", ">", "$")

      >>> OG.set_precedence("$", "<", "i")
      >>> OG.set_precedence("$", "<", "+")
      >>> OG.set_precedence("$", "<", "*")

      >>> OG.make_precedence_functions()

      >>> OG.get_entry("$+i     * +    $", -1)
      '$ < + < i > * > + > $'
    """

    output = last_symbol = w[0]
    for i in range(1, len(w)):
      # Ignoramos los espacios
      if w[i] == " ": continue

      if self.f[last_symbol] < self.g[w[i]]: output += " < "
      elif self.f[last_symbol] > self.g[w[i]]: output += " > "
      else: output += " = "

      last_symbol = w[i]
      output += "\033[1;5;7m"*(index == i) + w[i] + "\033[0m"*(index == i)
    return output

  def print_step(self, stack: List[str], w: str, index: int, action: str, l: int):
    """
      Imprime un paso en el proceso de parsear.
    """
    stack_str = ""
    for s in stack: stack_str += s + " "

    entry = self.get_entry(w, index)

    print(stack_str.ljust(l), entry.ljust(int(2.5*l + 12)), action)

  def verify_input(self, w: str):
    """
      Verifica que los simbolos en una palabra sean terminales y comparables entre si.
    """
    w = "$" + w.replace(" ", "") + "$"

    if len(w) == 0: return 

    # Verificamos que todos los simbolos son terminales y comparables con el simbolo anterior.
    for i in range(1, len(w)):
      if not w[i] in self.Sigma:
        if not w[i] in self.N:
          raise Exception(
            f'El simbolo \033[1;3m{w[0]}\033[0m no es parte de la gramatica.'
          )
        raise Exception(
          f'El simbolo \033[1;3m{w[0]}\033[0m es no-terminal.'
        )

      if not (w[i-1] in self.precedences and w[i] in self.precedences[w[i-1]]):
        raise Exception(
          f'No existe una relacion de precedencia entre los simbolos \033[1;3m{w[i-1]}\033[0m y \033[1;3m{w[i]}\033[0m.'
        )

  def parse(self, w: str):
    """
      Parsea una entrada.

      INPUTS:
        w: str  ->  Entrada a procesar

      >>> OG = OperatorGrammar()
      >>> OG.make_rule("S E")
      'S -> E '
      >>> OG.make_rule("E E + E + E")
      'E -> E + E + E '
      >>> OG.make_rule("E E * E")
      'E -> E * E '
      >>> OG.make_rule("E i")
      'E -> i '
      >>> OG.set_init("S")
      >>> OG.set_precedence("i", ">", "$")
      >>> OG.set_precedence("i", ">", "+")
      >>> OG.set_precedence("i", ">", "*")

      >>> OG.set_precedence("+", "<", "i")
      >>> OG.set_precedence("+", "=", "+")
      >>> OG.set_precedence("+", "<", "*")
      >>> OG.set_precedence("+", ">", "$")

      >>> OG.set_precedence("*", "<", "i")
      >>> OG.set_precedence("*", ">", "+")
      >>> OG.set_precedence("*", ">", "*")
      >>> OG.set_precedence("*", ">", "$")

      >>> OG.set_precedence("$", "<", "i")
      >>> OG.set_precedence("$", "<", "+")
      >>> OG.set_precedence("$", "<", "*")

      >>> OG.make_precedence_functions()

      >>> OG.parse("i+i+i*i")
      PILA               ENTRADA                                       ACCION
      $                  $ < [1;5;7mi[0m > + < i > + < i > * < i > $         Leer i
      $ i                $ < i > [1;5;7m+[0m < i > + < i > * < i > $         Reducir:  E -> i 
      $ E                $ < [1;5;7m+[0m < i > + < i > * < i > $             Leer +
      $ E +              $ < + < [1;5;7mi[0m > + < i > * < i > $             Leer i
      $ E + i            $ < + < i > [1;5;7m+[0m < i > * < i > $             Reducir:  E -> i 
      $ E + E            $ < + = [1;5;7m+[0m < i > * < i > $                 Leer +
      $ E + E +          $ < + = + < [1;5;7mi[0m > * < i > $                 Leer i
      $ E + E + i        $ < + = + < i > [1;5;7m*[0m < i > $                 Reducir:  E -> i 
      $ E + E + E        $ < + = + < [1;5;7m*[0m < i > $                     Leer *
      $ E + E + E *      $ < + = + < * < [1;5;7mi[0m > $                     Leer i
      $ E + E + E * i    $ < + = + < * < i > [1;5;7m$[0m                     Reducir:  E -> i 
      $ E + E + E * E    $ < + = + < * > [1;5;7m$[0m                         Reducir:  E -> E * E 
      $ E + E + E        $ < + = + > [1;5;7m$[0m                             Reducir:  E -> E + E + E 
      $ E                $ = $                                                 Reducir:  S -> E 
      $ S                $ = $                                                 [1;3;36mACEPTAR[0m

      >>> OG.parse("i**i")
      PILA         ENTRADA                        ACCION
      $            $ < [1;5;7mi[0m > * > * < i > $      Leer i
      $ i          $ < i > [1;5;7m*[0m > * < i > $      Reducir:  E -> i 
      $ E          $ < [1;5;7m*[0m > * < i > $          Leer *
      $ E *        $ < * > [1;5;7m*[0m < i > $          [1;3;31mRECHAZAR[0m. No se puede reducir E * 

      >>> OG.parse("")
      Traceback (most recent call last):
      ...
      Exception: No existe una relacion de precedencia entre los simbolos [1;3m$[0m y [1;3m$[0m.
    """

    # Verificamos que el input sea valida 
    self.verify_input(w)
    
    # Pila de simbolos
    stack = ["$"]
    # Agregamos $ en los extremos de la entrada
    w = "$" + w.replace(" ", "") + "$"
    l = 2 * len(w)
    print("PILA".ljust(l), "ENTRADA".ljust(int(2.5*l)), "ACCION")

    # Obtenemos el primer indice que no corresponda a espacio en la entrada
    index = 1
    while w[index] == " ": index += 1
    e = w[index]

    while True:
      if not e in self.Sigma:
        raise Exception(
          f'El simbolo \033[1;3m{e}\033[0m no es parte de la gramatica.'
        )
      # Obtenemos el ultimo simbolo terminal de la pila
      p = stack[self.get_last_terminal(stack, len(stack)) ]

      if (p == "$") and (e == "$"):
        production = stack[1:]
        production.reverse()
        # Verificamos que llegamos al simbolo inicial.
        if (len(stack) == 2) and (stack[1] == self.S):
          self.print_step(stack, w, index, '\033[1;3;36mACEPTAR\033[0m', l)
          return 

        # Si no, vemos si los elementos actuales corresponden a una produccion.
        elif not tuple(production) in self.P_ref:
          right = tuple(production)
          self.print_step(stack, w, index, f'\033[1;3;31mRECHAZAR\033[0m No se puede reducir {right}', l)
          return

        else:
          rule = self.get_rule(tuple(production))
          self.print_step(stack, w, index, f'Reducir:  {rule}', l)
          stack = ["$", self.P_ref[tuple(production)]]
          continue

      if self.f[p] <= self.g[e]:
        # Agregamos el elemento a la pila y obtenemos el siguiente elemento terminal
        self.print_step(stack, w, index, "Leer " + e + "", l)
        stack.append(e)
        index += 1
        while w[index] == " ": index += 1
        e = w[index]

      else:
        # Sacamos de la pila todos los simbolo no-terminales y los almacenamos.
        stack_copy = stack.copy()
        rule = []
        j = self.get_last_terminal(stack, len(stack))
        for k in range(len(stack)-1, j, -1): rule.append(stack.pop())

        # Obtenemos el siguiente elemento terminal.
        x = stack.pop()
        rule.append(x)
        delete = 1

        # Mientras sigamos leyendo un simbolo no terminal o el elemento del tope
        # de la pila tenga mayor precedencia que el elemento que sacamos
        while (not self.is_terminal(stack[-1], True)) or \
            (self.f[stack[-1]] >= self.g[x]):
          
          rule.append(stack.pop())
          # Si el elemento es terminal, lo almacenamos en x 
          if self.is_terminal(rule[-1], True): 
            x = rule[-1]
            delete += 1

        # Si la regla actual no corresponde a ninguna produccion, le agregamos
        # el siguiente simbolo de la pila si es no-terminal
        while not (tuple(rule) in self.P_ref or self.is_terminal(stack[-1], True)):
          rule.append(stack.pop())

        production = rule.copy()
        production.reverse()
        production = tuple(production)
        right = ""
        for t in production: right += t + " "

        if not production in self.P_ref:
          self.print_step(stack_copy, w, index, f'\033[1;3;31mRECHAZAR\033[0m. No se puede reducir {right}', l)
          return
        
        production_str = self.get_rule(production)
        self.print_step(stack_copy, w, index, f'Reducir:  {production_str}', l)
        stack.append(self.P_ref[production])
        w = w[:index-delete] + w[index:]
        index -= 1


def syntax_error():
  """
    Imprime un error de sintaxis y muestra la forma correcta de usarla.

    >>> syntax_error()
    \033[1;31mError:\033[0m Sintaxis invalida:
    Ejecute 
      \033[1mRULE\033[0m <\033[4mNO-TERMINAL\033[0m> [<\033[4mSIMBOLO\033[0m> ...]
      \033[1mINIT\033[0m <\033[4mNO-TERMINAL\033[0m>
      \033[1mPREC\033[0m <\033[4mNO-TERMINAL\033[0m> <\033[4mOPERATION\033[0m> <\033[4mNO-TERMINAL\033[0m>
      \033[1mBUILD\033[0m
      \033[1mPARSE\033[0m [<\033[4mSTRING\033[0m> ...]
      \033[1mSALIR\033[0m
    <BLANKLINE>
  """
  print(
    '\033[1;31mError:\033[0m Sintaxis invalida:\n' + \
    'Ejecute \n' + \
    '  \033[1mRULE\033[0m <\033[4mNO-TERMINAL\033[0m> [<\033[4mSIMBOLO\033[0m> ...]\n' + \
    '  \033[1mINIT\033[0m <\033[4mNO-TERMINAL\033[0m>\n' + \
    '  \033[1mPREC\033[0m <\033[4mNO-TERMINAL\033[0m> <\033[4mOPERATION\033[0m> <\033[4mNO-TERMINAL\033[0m>\n' + \
    '  \033[1mBUILD\033[0m\n' + \
    '  \033[1mPARSE\033[0m [<\033[4mSTRING\033[0m> ...]\n' + \
    '  \033[1mSALIR\033[0m\n'
  )

def RULE(command: str, OG: OperatorGrammar):
  try:
    production = OG.make_rule(command[5:])
    print(f'Regla \033[1;3m{production}\033[0m agregada a la gramatica.')
  except Exception as e:
    print("\033[1;31mError:\033[0m ", e)

def INIT(command: str, OG: OperatorGrammar):
  tokens = command.split()

  if len(tokens) != 2: 
    syntax_error()
    return

  try:
    OG.set_init(tokens[1])
    print(f'Se establecio \033[1;3m{tokens[1]}\033[0m como simbolo inicial de la gramatica.')
  except Exception as e:
    print("\033[1;31mError:\033[0m ", e)

def PREC(command: str, OG: OperatorGrammar):
  tokens = command.split()

  if len(tokens) != 4: 
    syntax_error()
    return

  try:
    OG.set_precedence(tokens[1], tokens[2], tokens[3])
    print(f'Se establecio la relacion de precedencia \033[1;3m{tokens[1]} {tokens[2]} {tokens[3]}\033[0m.')
  except Exception as e:
    print("\033[1;31mError:\033[0m ", e)

def BUILD(OG: OperatorGrammar):
  if OG.S == None:
    print("\033[1;31mError:\033[0m No se ha definido un estado inicial.")
    return

  try:
    OG.make_precedence_functions()
    print('Analizador sintaction construido.')
    print('Los valores de f son:')
    Sigma = list(OG.Sigma)
    Sigma.sort()
    for s in Sigma:
      print(f'  f({s}) = {OG.f[s]}')
    print('Los valores de g son:')
    for s in Sigma:
      print(f'  g({s}) = {OG.g[s]}')
  except Exception as e:
    print("\033[1;31mError:\033[0m ", e)

def PARSE(command: str, OG: OperatorGrammar):
  if not OG.builded:
    print("\033[1;31mError:\033[0m Aun no se ha construido el analizador sintactico.")
    return

  try:
    OG.parse(command[6:])
  except Exception as e:
    print("\033[1;31mError:\033[0m ", e)

def main(input = input):
  """
    >>> index = [-1]
    >>> def fake_input(ignore: str, index = index):
    ...   r = [
    ...     "RULE E E + E",
    ...     "RULE E E + E",
    ...     "RULE E E * E",
    ...     "RULE E E E",
    ...     "RULE E n",
    ...     "INIT e",
    ...     "INIT E",
    ...     "PREC n > +",
    ...     "PREC n > *",
    ...     "PREC n > $",
    ...     "PREC + < n",
    ...     "PREC + > +",
    ...     "PREC + < *",
    ...     "PREC + > $",
    ...     "PREC * < n",
    ...     "PREC * > +",
    ...     "PREC * > *",
    ...     "PREC * > $",
    ...     "PREC $ < n",
    ...     "PREC $ < +",
    ...     "PREC $ < *",
    ...     "PARSE n + n * n",
    ...     "BUILD",
    ...     "PARSE n + n * n",
    ...     "PARSE n + * n",
    ...     "PARSE n",
    ...     "PARSE a + b * c",
    ...     "PARSE n + n * c",
    ...     "PARSE n + n n",
    ...     "PARSE n + n E",
    ...     "PARSE",
    ...     "SALIR", 
    ...   ]
    ...   index[0] += 1
    ...   return r[index[0]]

    >>> main(fake_input)
    Generador de analizadores de [3mGramaticas de Operadores[0m.
    <BLANKLINE>
    [1mSYNOPSIS[0m
      [1mRULE[0m <[4mNO-TERMINAL[0m> [<[4mSIMBOLO[0m> ...]
      [1mINIT[0m <[4mNO-TERMINAL[0m>
      [1mPREC[0m <[4mNO-TERMINAL[0m> <[4mOPERATION[0m> <[4mNO-TERMINAL[0m>
      [1mBUILD[0m
      [1mPARSE[0m [<[4mSTRING[0m> ...]
      [1mSALIR[0m
    <BLANKLINE>
    Para obtener un analisis de prueba unitarias y cobertura, instale la libreria [3mcoverage[0m y ejecute
      $ coverage run OperatorGrammarAnalyzer.py --test && coverage annotate && coverage report
    <BLANKLINE>
    Regla [1;3mE -> E + E [0m agregada a la gramatica.
    Regla [1;3mE -> E + E [0m agregada a la gramatica.
    Regla [1;3mE -> E * E [0m agregada a la gramatica.
    [1;31mError:[0m  La regla contiene dos simbolos no-terminales [1;3mE[0m y [1;3mE[0m seguidos. 
    Recuerde que una [3mGramatica de Operadores[0m no debe contener dos simbolos no-terminales seguidos.
    Regla [1;3mE -> n [0m agregada a la gramatica.
    [1;31mError:[0m  El simbolo [1;3me[0m es terminal. 
    El simbolo inicial debe ser no-terminal.
    Se establecio [1;3mE[0m como simbolo inicial de la gramatica.
    Se establecio la relacion de precedencia [1;3mn > +[0m.
    Se establecio la relacion de precedencia [1;3mn > *[0m.
    Se establecio la relacion de precedencia [1;3mn > $[0m.
    Se establecio la relacion de precedencia [1;3m+ < n[0m.
    Se establecio la relacion de precedencia [1;3m+ > +[0m.
    Se establecio la relacion de precedencia [1;3m+ < *[0m.
    Se establecio la relacion de precedencia [1;3m+ > $[0m.
    Se establecio la relacion de precedencia [1;3m* < n[0m.
    Se establecio la relacion de precedencia [1;3m* > +[0m.
    Se establecio la relacion de precedencia [1;3m* > *[0m.
    Se establecio la relacion de precedencia [1;3m* > $[0m.
    Se establecio la relacion de precedencia [1;3m$ < n[0m.
    Se establecio la relacion de precedencia [1;3m$ < +[0m.
    Se establecio la relacion de precedencia [1;3m$ < *[0m.
    [1;31mError:[0m Aun no se ha construido el analizador sintactico.
    Analizador sintaction construido.
    Los valores de f son:
      f($) = 0
      f(*) = 4
      f(+) = 2
      f(n) = 4
    Los valores de g son:
      g($) = 0
      g(*) = 3
      g(+) = 1
      g(n) = 5
    PILA           ENTRADA                             ACCION
    $              $ < [1;5;7mn[0m > + < n > * < n > $           Leer n
    $ n            $ < n > [1;5;7m+[0m < n > * < n > $           Reducir:  E -> n 
    $ E            $ < [1;5;7m+[0m < n > * < n > $               Leer +
    $ E +          $ < + < [1;5;7mn[0m > * < n > $               Leer n
    $ E + n        $ < + < n > [1;5;7m*[0m < n > $               Reducir:  E -> n 
    $ E + E        $ < + < [1;5;7m*[0m < n > $                   Leer *
    $ E + E *      $ < + < * < [1;5;7mn[0m > $                   Leer n
    $ E + E * n    $ < + < * < n > [1;5;7m$[0m                   Reducir:  E -> n 
    $ E + E * E    $ < + < * > [1;5;7m$[0m                       Reducir:  E -> E * E 
    $ E + E        $ < + > [1;5;7m$[0m                           Reducir:  E -> E + E 
    $ E            $ = [1;5;7m$[0m                               [1;3;36mACEPTAR[0m
    PILA         ENTRADA                        ACCION
    $            $ < [1;5;7mn[0m > + < * < n > $          Leer n
    $ n          $ < n > [1;5;7m+[0m < * < n > $          Reducir:  E -> n 
    $ E          $ < [1;5;7m+[0m < * < n > $              Leer +
    $ E +        $ < + < [1;5;7m*[0m < n > $              Leer *
    $ E + *      $ < + < * < [1;5;7mn[0m > $              Leer n
    $ E + * n    $ < + < * < n > [1;5;7m$[0m              Reducir:  E -> n 
    $ E + * E    $ < + < * > [1;5;7m$[0m                  [1;3;31mRECHAZAR[0m. No se puede reducir * E 
    PILA   ENTRADA         ACCION
    $      $ < [1;5;7mn[0m > $       Leer n
    $ n    $ < n > [1;5;7m$[0m       Reducir:  E -> n 
    $ E    $ = [1;5;7m$[0m           [1;3;36mACEPTAR[0m
    [1;31mError:[0m  El simbolo [1;3m$[0m no es parte de la gramatica.
    [1;31mError:[0m  El simbolo [1;3m$[0m no es parte de la gramatica.
    [1;31mError:[0m  No existe una relacion de precedencia entre los simbolos [1;3mn[0m y [1;3mn[0m.
    [1;31mError:[0m  El simbolo [1;3m$[0m es no-terminal.
    [1;31mError:[0m  No existe una relacion de precedencia entre los simbolos [1;3m$[0m y [1;3m$[0m.
    Hasta luego!
  """
  OG = OperatorGrammar()

  print(
    'Generador de analizadores de \033[3mGramaticas de Operadores\033[0m.\n\n' + \
    '\033[1mSYNOPSIS\033[0m\n' +\
    '  \033[1mRULE\033[0m <\033[4mNO-TERMINAL\033[0m> [<\033[4mSIMBOLO\033[0m> ...]\n' + \
    '  \033[1mINIT\033[0m <\033[4mNO-TERMINAL\033[0m>\n' + \
    '  \033[1mPREC\033[0m <\033[4mNO-TERMINAL\033[0m> <\033[4mOPERATION\033[0m> <\033[4mNO-TERMINAL\033[0m>\n' + \
    '  \033[1mBUILD\033[0m\n' + \
    '  \033[1mPARSE\033[0m [<\033[4mSTRING\033[0m> ...]\n' + \
    '  \033[1mSALIR\033[0m\n\n' + \
    'Para obtener un analisis de prueba unitarias y cobertura, instale la libreria \033[3mcoverage\033[0m y ejecute\n' + \
    '  $ coverage run OperatorGrammarAnalyzer.py --test && coverage annotate && coverage report\n'
  )

  while True:
    command = input("$> ")

    if not command: 
      syntax_error()

    else:
      # La accion es el primer argumento del comando
      action = command.split()[0]
      if action == "RULE": RULE(command, OG)
      elif action == "INIT": INIT(command, OG)
      elif action == "PREC": PREC(command, OG)
      elif action == "BUILD": BUILD(OG)
      elif action == "PARSE": PARSE(command, OG)
      elif action == "SALIR": print("Hasta luego!"); break
      else: syntax_error()


if __name__ == "__main__":
  from sys import argv

  if (len(argv) == 2) and (argv[1] == "--test"):
    import doctest
    doctest.testmod(verbose=False)
  elif len(argv) == 1:
    main()
  else:
    raise Exception(f'Argumentos {argv[1:]} invalidos.')