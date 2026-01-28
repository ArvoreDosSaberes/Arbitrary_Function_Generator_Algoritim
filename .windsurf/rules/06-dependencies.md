---
trigger: model_decision
description: Deve ser sempre utilizada quando for manipular dependências, package.json, requirements.txt, CMakeList.txt entre outros.
---
## **Novas Dependências**

* Sempre que for necessário adicionar **novas dependências** ao projeto, certifique-se de que o processo de instalação e integração seja **verificado e controlado**.
* Antes da instalação:

  1. **Verifique a compatibilidade** da nova dependência com as versões já existentes no projeto (bibliotecas, frameworks, runtime e compiladores).
  2. Avalie **possíveis conflitos de versão** e, se necessário, realize **downgrade ou sincronização** dos pacotes para garantir estabilidade.
  3. Em projetos Node.js, utilize o comando:

     ```bash
     npm install
     ```

     após adicionar ou atualizar dependências, para assegurar que todas foram instaladas corretamente e estão registradas no `package-lock.json`.
* Após a instalação:

  * Execute os **testes unitários e de integração** para validar a estabilidade do sistema.
  * Atualize o arquivo de **documentação técnica** (`README.md` ou `CHANGELOG.md`) descrevendo a dependência adicionada, sua versão e propósito.
  * Caso o projeto utilize containers, atualize também o **Dockerfile** e o **package-lock.json** (ou `requirements.txt` no caso de Python).
    <<<<<<< HEAD
* Sempre mantenha as dependências **atualizadas, compatíveis e rastreáveis**, garantindo a reprodutibilidade do ambiente de desenvolvimento e produção.

=======

* Sempre mantenha as dependências **atualizadas, compatíveis e rastreáveis**, garantindo a reprodutibilidade do ambiente de desenvolvimento e produção.

>>>>>>> main
>>>>>>>
>>>>>>
>>>>>
>>>>
>>>
>>
