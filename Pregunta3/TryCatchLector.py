 
class TryCatchLector:
  """
    Reconocedor de instrucciones try-catch-finally.
  """

  def __init__(self):
    self.entry = []
    self.stack = []
    self.l = -1

  def get_stack_str(self) -> str:
    """
      Retorna una representacion de la pila.
    """
    output = ""
    for s in self.stack:
      output += f'({s[0]}, {s[1]}) '
    return output

  def get_entry_str(self) -> str:
    """
      Retorna una representacion de la entrada
    """
    output = ""
    for e in self.entry:
      output += f'{e} '
    return output

  def print_action(self, action: str):
    """
      Imprime una accion del reconocedor.
    """
    print(self.get_stack_str().ljust(int(3/4*self.l)), self.get_entry_str().ljust(self.l+4), action)

  def shift(self, symbol: str) -> bool:
    """ 
      Verificamos que el siguiente simbolo de la entrada sea el mismo que al
      que se le esta haciendo shift.
    """
    if self.entry[0] == symbol:
      self.print_action(f'Leer')
      if symbol[:6] == 'instr_': self.stack.append([symbol[6:], ''])
      self.entry.pop(0)
      return True 
    return False

  def LA(self) -> str:
    """ Lookahead. """
    if len(self.entry) > 0: return self.entry[0]
    else: return ''

  def S(self) -> str:
    """
      S  ->  I  { S.tipo <- I.tipo }  $
    """
    if self.LA() == 'try' or self.LA()[:6] == 'instr_':
      self.print_action(f'Expandir \033[1mS -> I\033[0m')
      self.stack.append(['', ''])
      self.I()
      if not self.shift('$'):
        raise Exception('No se pudo reducir S -> I $')
      self.print_action(f'\033[1;3;36mACEPTAR\033[0m')
      return self.stack.pop()[0]
    else:
      raise Exception('No se pudo expandir S')

  def I(self):
    """
      I  ->  E  { R.in <- E.tipo }  R  { I.tipo <- R.tipo }
    """
    if self.LA() == 'try' or self.LA()[:6] == 'instr_':
      self.print_action(f'Expandir \033[1mI -> E R\033[0m')
      self.stack.append(['', ''])
      self.E()
      self.stack.append(['', self.stack.pop()[0]])
      self.R()
      symbol = self.stack.pop()
      self.stack[-1][0] = symbol[0]
    else:
      raise Exception('No se pudo expandir I')

  def R(self):
    """
      R  ->  ; I  { R0.in <- I.tipo }  R0  { R.tipo <- R0.tipo }
          |  { R.tipo <- R.in }
    """
    if self.LA() == ';':
      # R  ->  ; I  { R0.in <- I.tipo }  R0  { R.tipo <- R0.tipo }
      self.print_action(f'Expandir \033[1mR -> ; I R\033[0m')
      if not self.shift(';'):
        raise Exception('No se pudo reducir R -> ; I R $')
      self.stack.append(['', ''])
      self.I()
      self.stack.append(['', self.stack.pop()[0]])
      self.R()
      symbol = self.stack.pop()
      self.stack[-1][0] = symbol[0]

    elif self.LA() in {'$', 'catch', 'finally'}:
      # R  ->  { R.tipo <- R.in }
      self.print_action(f'Expandir \033[1mR -> \033[0m')
      self.stack[-1][0] = self.stack[-1][1]

    else:
      raise Exception('No se pudo expandir R')

  def E(self):
    """
      E  ->  try I0 catch  { I1.in <- IO.tipo }  
                I1  { F.in <- Either I1.in I1.tipo}
                F  { E.tipo <- F.tipo }
          |  instr  { E.tipo <- instr.tipo }  
    """
    if self.LA() == 'try':
      # E  ->  try I0 catch  { I1.in <- IO.tipo }  
      #           I1  { F.in <- Either I1.in I1.tipo}
      #           F  { E.tipo <- F.tipo }
      self.print_action(f'Expandir \033[1mE -> try I catch I F\033[0m')
      self.shift('try')
      self.stack.append(['', ''])
      self.I()
      if not self.shift('catch'):
        raise Exception('No se pudo reducir E -> try I catch I F')
      self.stack.append(['', self.stack.pop()[0]])
      self.I()
      symbol = self.stack.pop()
      self.stack.append(['', f'Either {symbol[1]} {symbol[0]}'])
      self.F()
      symbol = self.stack.pop()
      self.stack[-1][0] = symbol[0]

    elif self.LA()[:6] == 'instr_':
      # E  ->  instr  { E.tipo <- instr.tipo }
      self.print_action(f'Expandir \033[1mE -> instr\033[0m')
      self.shift(self.LA())
      symbol = self.stack.pop()[0]
      self.stack[-1][0] = symbol

    else:
      raise Exception('No se pudo expandir E')

  def F(self):
    """
      F  ->  finally E  { F.tipo <- E.tipo }
          |  { F.tipo <- F.in }
    """
    if self.LA() == 'finally':
      # F  ->  finally E  { F.tipo <- E.tipo }
      self.print_action(f'Expandir \033[1mF -> finally E\033[0m')
      self.shift('finally')
      self.stack.append(['', ''])
      self.E()
      symbol = self.stack.pop()[0]
      self.stack[-1][0] = symbol
    
    elif self.LA() in {'$', ';', 'catch'}:
      # F  ->  { F.tipo <- F.in }
      self.print_action(f'Expandir \033[1mF -> \033[0m')
      self.stack[-1][0] = self.stack[-1][1]

    else:
      raise Exception('No se pudo expandir F')

  def parse(self, w: str) -> str:
    self.entry = w.split() + ['$']
    self.l = len(self.get_entry_str())
    print('\033[1mPILA'.ljust(int(3/4*self.l) + 4), 'ENTRADA'.ljust(self.l+4), 'ACCION\033[0m')

    try:
      return self.S()
    except Exception as e:
      self.print_action(f'\033[1;3;31mRECHAZAR\033[0m')

def syntax_error():
  """
    Imprime un error de sintaxis y muestra la forma correcta de usarla.
  """
  print(
    '\033[1;31mError:\033[0m Sintaxis invalida:\n' + \
    'Ejecute \n' + \
    '  \033[1mPARSE\033[0m [<\033[4mSTRING\033[0m> ...]\n' + \
    '  \033[1mSALIR\033[0m\n'
  )

def main(input = input):
  tcl = TryCatchLector()

  print(
    'Reconocedor recursivo descendente de instrucciones \033[3mtry-catch-finally\033[0m.\n' + \
    'El tipo de retorno de una instruccion atomica \033[3minstr\033[0m debe indicarse luego ' + \
    'de la instruccion separadados por \'_\'.\nPor ejemplo \033[1minstr_X\033[0m sera de tipo ' + \
    '\033[1mX\033[0m.\n'
  )

  while True:
    command = input("$> ")

    if not command: 
      syntax_error()

    else:
      # La accion es el primer argumento del comando
      action = command.split()[0]
      if action == "PARSE": 
        tipo = tcl.parse(command[6:])
        if tipo != None:
          print(f'\nTipo de la instruccion: \033[1;3m{tipo}\033[0m')
      elif action == "SALIR": print("Hasta luego!"); break
      else: syntax_error()

if __name__ == '__main__': main()
  
  