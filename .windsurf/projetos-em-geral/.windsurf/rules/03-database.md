---
<<<<<<< HEAD
trigger: always_on
=======
trigger: model_decision
description: Criação e manutenção do banco de dados, não importa qual banco de dados seja escolhido.
>>>>>>> main
---

## **Banco de Dados**

* Utilize **MySQL** ou **MariaDB** com o **mecanismo de armazenamento InnoDB** como padrão, salvo indicação contrária.
* A aplicação deve, em **cada nova versão**, verificar automaticamente a **existência do banco e das tabelas** durante a inicialização.

  * Caso o banco ou tabelas não existam, o sistema deve criá-los automaticamente.
* Mantenha um **controle de versão do banco de dados**:

  * Registre o histórico de atualizações estruturais (migrations).
  * Ao iniciar, o sistema deve **detectar a versão atual** e aplicar **atualizações incrementais** para alinhar à nova versão do esquema.
* As operações de verificação e atualização devem ser **idempotentes**, garantindo que possam ser executadas múltiplas vezes sem efeitos colaterais.

---

## **Scripts de Banco de Dados**

* Todos os **scripts de manutenção e migração** devem ser armazenados na pasta:

  ```
  /scripts/db/
  ```

  localizada na raiz do projeto.
* Os scripts devem identificar **claramente o contexto de execução**:

  * Use o **usuário root** apenas para operações administrativas (criação de usuários e bancos).
  * Use o **usuário proprietário do banco de dados** para operações rotineiras (inserções, migrações e consultas).
* Ao criar um novo usuário de banco de dados, o script deve também **criar seu respectivo schema/database**, conforme as orientações do sistema.
* O sistema deve implementar rotinas automáticas de **verificação na inicialização**, garantindo:

  * Existência do banco e tabelas obrigatórias;
  * Aplicação de scripts de migração pendentes;
<<<<<<< HEAD
  * Consistência entre a versão do sistema e a versão registrada do banco.

=======
  * Consistência entre a versão do sistema e a versão registrada do banco.
>>>>>>> main
