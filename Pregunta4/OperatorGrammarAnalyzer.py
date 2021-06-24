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
    self.f_nodes = {}
    self.g_nodes = {}
    self.f = {}
    self.g = {}
    # source_nodes guarda los nodos fuente, cuyo grado interno es 0
    self.source_nodes_f = set()
    self.source_nodes_g = set()
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
    """

    if len(symbol) > 1:
      raise Exception(
        f'El simbolo \033[1;3m{symbol}\033[0m no es valido. Por favor, utilice ' + \
        'caracteres de la tabla ASCII que esten en el rango de valores [33, 127), ' + \
        'excepto el 36 que corresponde al caracter \033[1;3m$\033[0m.'
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
    if left in self.equiv_graph:
      if right in self.equiv_graph[left]:
        print(
          '\033[1;35mWarning:\033[0m Ya se definio la relacion de precedencia ' + \
          f'\033[1;3m=\033[0m entre \033[1;3m{left}\033[0m y \033[1;3m{right}\033[0m.'
        )
        self.equiv_graph[left].discard(right)
        self.equiv_graph[right].discard(left)

  
    # Si el operador es =, agregamos a right y left a la misma clase de equivalencia.
    if op == '=':

      # Agregamos la relacion de equivalencia right -> left
      if right in self.equiv_graph: self.equiv_graph[right].add(left)
      else: self.equiv_graph[right] = set([left])
      # Agregamos la relacion de equivalencia left -> right
      if left in self.equiv_graph: self.equiv_graph[left].add(right)
      else: self.equiv_graph[left] = set([right])

    else: 
      # Establecemos la precedencia
      if left in self.precedences: self.precedences[left][right] = op
      else: self.precedences[left] = {right: op}

    self.Sigma.add(left)
    self.Sigma.add(right)

  def dfs(self, symbol: str, visited: Set[str], count: int) -> Set[str]:
    """
      Aplicamos DFS para crear la clases de equivalencia del simbolo.

      INPUTS:
        symbol: str       ->  Simbolo (nodo) actual.
        visited: Set[str] ->  Nodos ya visitados.
        count: int        ->  Clase de equivalencia actual.
      OUTPUT:
        Set[str]  ->  Nodos visitados luego de esta iteracion.

      >>> OG = OperatorGrammar()
      >>> OG.set_precedence("a", "=", "b")
      >>> OG.set_precedence("b", "=", "c")
      >>> OG.set_precedence("b", "=", "d")
      >>> OG.set_precedence("c", "=", "d")
      >>> OG.set_precedence("d", "=", "e")

      >>> s = list(OG.dfs("a", set(), 0))
      >>> s.sort()
      >>> s
      ['a', 'b', 'c', 'd', 'e']

      >>> [(t, OG.equivs[t]) for t in s]
      [('a', 0), ('b', 0), ('c', 0), ('d', 0), ('e', 0)]
    """

    self.equivs[symbol] = count
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
      >>> OG.set_precedence("b", "=", "c")
      >>> OG.set_precedence("f", "=", "e")
      >>> OG.set_precedence("g", "=", "h")
      >>> OG.set_precedence("i", "=", "h")
      >>> OG.make_equiv_graph()

      >>> OG.equivs["$"] != OG.equivs["a"]
      True
      >>> (OG.equivs["a"] == OG.equivs["b"]) and (OG.equivs["b"] == OG.equivs["c"])
      True
      >>> OG.equivs["c"] != OG.equivs["e"]
      True
      >>> OG.equivs["e"] == OG.equivs["f"]
      True
      >>> OG.equivs["f"] != OG.equivs["g"]
      True
      >>> (OG.equivs["g"] == OG.equivs["h"]) and (OG.equivs["h"] == OG.equivs["i"])
      True
    """

    # Limpiamos equivs por si guardaba valores anteriores
    self.equivs = {}

    visited = set()
    count = 0

    # Aplicamos DFS para enumerar las clases de equivalencia
    for symbol in self.Sigma:
      if not symbol in visited:
        visited = self.dfs(symbol, visited, count)
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

      >>> OG.g_nodes[OG.equivs["c"]] == set([OG.equivs["f"]])
      True
      >>> OG.f_nodes[OG.equivs["f"]] == set([OG.equivs["z"], OG.equivs["q"]])
      True
    """

    # Creamos las clases de equivalencia
    self.make_equiv_graph()

    # Limpiamos f_nodes, g_nodes y source_nodes por si guardan valores anteriores
    self.f_nodes = {}
    self.g_nodes = {}
    self.source_nodes_f = set()
    self.source_nodes_g = set()

    for node in self.equivs: 
      self.source_nodes_f.add(self.equivs[node])
      self.source_nodes_g.add(self.equivs[node])

    # Creamos las relaciones de las clases de equivalencia
    for left in self.Sigma:
      if not left in self.precedences: continue

      for right in self.precedences[left]:
        
        # Si left < right, entonces  [g_right] -> [f_left] 
        if self.precedences[left][right] == '<':
          if self.equivs[right] in self.g_nodes: 
            self.g_nodes[self.equivs[right]].add(self.equivs[left])
          else: 
            self.g_nodes[self.equivs[right]] = set([self.equivs[left]])

          self.source_nodes_f.discard(self.equivs[left])

        # En caso contrario (si es >), entonces  [f_left] -> [g_right]
        else:
          if self.equivs[left] in self.f_nodes: 
            self.f_nodes[self.equivs[left]].add(self.equivs[right])
          else: 
            self.f_nodes[self.equivs[left]] = set([self.equivs[right]])

          self.source_nodes_g.discard(self.equivs[right])

  def precedences_error(self, path: List[Tuple[int, bool]]):
    """
      Obtenemos el ciclo de precedencia y lo reportamos.
    """
    for i in range(len(path)-2, -1, -1): 
      if path[i] == path[-1]:
        index = i
        break 

    error = "Ha ocurrido un ciclo en las precedencias.\n" + \
      "A continuacion reportaremos las clases de equivalencias relacionada:\n\n"

    # Agrupamos los simbolos de la misma clase
    equiv_classes = [set() for _ in range(len(self.equivs))]
    for s in self.Sigma:
      equiv_classes[self.equivs[s]].add(s)

    # Creamos un string que representa el ciclo
    for i in range(index, len(path)-1):
      error += str(equiv_classes[path[i][0]]) + " -> "
    error += str(equiv_classes[path[-1][0]])

    # Reportamos el error
    raise Exception(error)

  def get_max_path(self, path: List[Tuple[int, bool]], path_set: Set[Tuple[int, bool]]) -> int:
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
      >>> OG.get_max_path([(OG.equivs["i"], False)], set([(OG.equivs["i"], False)]))
      5
    """
 
    node = path[-1][0]
    is_f = path[-1][1]

    # Si ya calculamos el maximo camino, lo retornamos.
    if is_f and node in self.f: return self.f[node] 
    if not is_f and node in self.g: return self.g[node]

    # Obtenemos el maximo valor de los hijos.
    max_value = 0

    if is_f:
      # Si no tiene nodos hijos, entonces su valor es 0
      if not node in self.f_nodes:
        self.f[node] = 0
        return 0 

      # Por cada hijo del nodo actual.
      for g_node in self.f_nodes[node]:

        # Si el hijo ya esta en el camino actual, entonces hay un ciclo
        if (g_node, False) in path_set: 
          self.precedences_error(path + [(g_node, False)])

        path.append((g_node, False))
        path_set.add((g_node, False))
        max_value = max(max_value, self.get_max_path(path, path_set))
        path_set.discard((g_node, False))
        path.pop()

      self.f[node] = max_value + 1
      return max_value + 1

    else:
      # Si no tiene nodos hijos, entonces su valor es 0
      if not node in self.g_nodes:
        self.g[node] = 0
        return 0 

      # Por cada hijo del nodo actual.
      for f_node in self.g_nodes[node]:

        # Si el hijo ya esta en el camino actual, entonces hay un ciclo
        if (f_node, True) in path_set: 
          self.precedences_error(path + [(f_node, True)])

        path.append((f_node, True))
        path_set.add((f_node, True))
        max_value = max(max_value, self.get_max_path(path, path_set))
        path_set.discard((f_node, True))
        path.pop()

      self.g[node] = max_value + 1
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
      ...   print("f(" + s + ") = ", OG.f[OG.equivs[s]])
      f($) =  0
      f(+) =  2
      f(*) =  4
      f(i) =  4
      >>> for s in ["$", "+", "*", "i"]:
      ...   print("g(" + s + ") = ", OG.g[OG.equivs[s]])
      g($) =  0
      g(+) =  1
      g(*) =  3
      g(i) =  5

      >>> OG.set_precedence("a", "=", "b")
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

    for f_node in self.source_nodes_f:
      path = [(f_node, True)]
      path_set = set([(f_node, True)])
      self.get_max_path(path, path_set)

    for g_node in self.source_nodes_g:
      path = [(g_node, False)]
      path_set = set([(g_node, False)])
      self.get_max_path(path, path_set)

    for s in self.Sigma:
      if not self.equivs[s] in self.f:
        path = [(self.equivs[s], True)]
        path_set = set([(self.equivs[s], True)])
        self.get_max_path(path, path_set)

      if not self.equivs[s] in self.g:
        path = [(self.equivs[s], False)]
        path_set = set([(self.equivs[s], False)])
        self.get_max_path(path, path_set)

    self.builded = True

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

  def get_entry(self, w: str) -> str:
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

      >>> OG.get_entry("$+i     * +    $")
      '$ < + < i > * > + > $'
    """

    output = last_symbol = w[0]
    for i in range(1, len(w)):
      # Ignoramos los espacios
      if w[i] == " ": continue

      if self.f[self.equivs[last_symbol]] < self.g[self.equivs[w[i]]]: output += " < "
      elif self.f[self.equivs[last_symbol]] > self.g[self.equivs[w[i]]]: output += " > "
      else: output += " = "

      last_symbol = w[i]
      output += w[i]
    return output

  def print_step(self, stack: List[str], w: str, action: str, l: int):
    """
      Imprime un paso en el proceso de parsear.
    """
    stack_str = ""
    for s in stack: stack_str += s + " "

    entry = self.get_entry(w)

    print(stack_str.ljust(l), entry.ljust(int(2.5*l)), action)

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
      $                  $ < i > + < i > + < i > * < i > $             Leer i
      $ i                $ < i > + < i > + < i > * < i > $             Reducir:  E -> i 
      $ E                $ < + < i > + < i > * < i > $                 Leer +
      $ E +              $ < + < i > + < i > * < i > $                 Leer i
      $ E + i            $ < + < i > + < i > * < i > $                 Reducir:  E -> i 
      $ E + E            $ < + = + < i > * < i > $                     Leer +
      $ E + E +          $ < + = + < i > * < i > $                     Leer i
      $ E + E + i        $ < + = + < i > * < i > $                     Reducir:  E -> i 
      $ E + E + E        $ < + = + < * < i > $                         Leer *
      $ E + E + E *      $ < + = + < * < i > $                         Leer i
      $ E + E + E * i    $ < + = + < * < i > $                         Reducir:  E -> i 
      $ E + E + E * E    $ < + = + < * > $                             Reducir:  E -> E * E 
      $ E + E + E        $ < + = + > $                                 Reducir:  E -> E + E + E 
      $ E                $ = $                                         Reducir:  S -> E 
      $ S                $ = $                                         Aceptar

      >>> OG.parse("i**i")
      PILA         ENTRADA                        ACCION
      $            $ < i > * > * < i > $          Leer i
      $ i          $ < i > * > * < i > $          Reducir:  E -> i 
      $ E          $ < * > * < i > $              Leer *
      $ E *        $ < * > * < i > $              Rechazar. No se puede reducir E * 

      >>> OG.parse("")
      PILA ENTRADA    ACCION
      $    $ = $      Rechazar. No se puede reducir ()
    """

    # Pila de simbolos
    stack = ["$"]
    # Agregamos $ en los extremos de la entrada
    w = "$" + w + "$"
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
          self.print_step(stack, w, 'Aceptar', l)
          return 

        # Si no, vemos si los elementos actuales corresponden a una produccion.
        elif not tuple(production) in self.P_ref:
          right = tuple(production)
          self.print_step(stack, w, f'Rechazar. No se puede reducir {right}', l)
          return

        else:
          rule = self.get_rule(tuple(production))
          self.print_step(stack, w, f'Reducir:  {rule}', l)
          stack = ["$", self.P_ref[tuple(production)]]
          continue

      if self.f[self.equivs[p]] <= self.g[self.equivs[e]]:
        # Agregamos el elemento a la pila y obtenemos el siguiente elemento terminal
        self.print_step(stack, w, "Leer " + e + "", l)
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
            (self.f[self.equivs[stack[-1]]] >= self.g[self.equivs[x]]):
          
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
          self.print_step(stack_copy, w, f'Rechazar. No se puede reducir {right}', l)
          return
        
        production_str = self.get_rule(production)
        self.print_step(stack_copy, w, f'Reducir:  {production_str}', l)
        stack.append(self.P_ref[production])
        w = w[:index-delete] + w[index:]
        index -= 1


def syntax_error():
  """
    Imprime un error de sintaxis y muestra la forma correcta de usarla.
  """
  print(
    '\033[1;31mError:\033[0m Sintaxis invalida:\n' + \
    'Ejecute \n' + \
    '\t\033[1mRULE\033[0m <\033[4mNO-TERMINAL\033[0m> [<\033[4mSIMBOLO\033[0m> ...]\n' + \
    '\t\033[1mINIT\033[0m <\033[4mNO-TERMINAL\033[0m>\n' + \
    '\t\033[1mPREC\033[0m <\033[4mNO-TERMINAL\033[0m> <\033[4mOPERATION\033[0m> <\033[4mNO-TERMINAL\033[0m>\n' + \
    '\t\033[1mBUILD\033[0m\n' + \
    '\t\033[1mPARSE\033[0m [<\033[4mSTRING\033[0m> ...]\n' + \
    '\t\033[1mSALIR\033[0m\n'
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
    for s in OG.Sigma:
      print(f'  f({s}) = {OG.f[OG.equivs[s]]}')
    print('Los valores de g son:')
    for s in OG.Sigma:
      print(f'  g({s}) = {OG.g[OG.equivs[s]]}')
  except Exception as e:
    print("\033[1;31mError:\033[0m ", e)

def PARSE(command: str, OG: OperatorGrammar):
  try:
    production = OG.make_rule(command[6:])
  except Exception as e:
    print("\033[1;31mError:\033[0m ", e)

def main(input = input):
  OG = OperatorGrammar()
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
  import doctest
  doctest.testmod(verbose=False)
  main()