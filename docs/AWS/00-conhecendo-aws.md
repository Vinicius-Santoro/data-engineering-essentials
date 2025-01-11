# **Cloud AWS**

## **Iniciando com AWS**

A AWS é um provedor Cloud utilizado globalmente. Sua história resumida se dá por:

- 2002: Amazon utilizada sua infraestrutura internamente.
- 2003: Dado sua infraestrutura excelente, surgiu a ideia de externalizar seus serviços.
- 2004: Lançaram o AWS SQS para o público.
- 2006: Expandiram a oferta com o S3 e EC2 para o público.
- Atualmente: Principal provedor cloud.

## **Infraestrutura Global da AWS**

### AWS Regiões
As regiões da AWS são locais físicos onde a AWS opera clusters de data centers,
identificadas por nomes como **us-east-1** e **eu-west-3**.

Ao escolher uma região para lançar um aplicativo, por exemplo, deve ser
considerado os seguintes fatores:

- Conformidade legal e governança de dados: Os dados não saem da região sem sua
permissão explícita.
- Proximidade dos clientes: Menor latência ao acessar os serviços.
- Serviços disponíveis na região: Nem todos os serviços e recursos estão
disponíveis em todas as regiões.
- Preços: Os custos variam entre regiões e podem ser consultados na página de
preços dos serviços.

### AWS Zonas de Disponibilidadae

Cada região da AWS possui de 3 a 6 Zonas de Disponibilidade (Availability Zones).
Cada zona é composta por um ou mais data centers, projetados com redundância em
energia, rede e conectividade.

Essa estrutura garante alta disponibilidade e resiliência: se uma zona de
disponibilidade for afetada por um desastre natural, a aplicação permanece
operacional devido à redundância da infraestrutura física.

<img src="./assets/aws-azs.png" alt="Exemplo: AWS Zona de Disponibilidade" width="40" height="40"/> 

### AWS Edge Locations / Points of Presence

A AWS possui mais de 400 Points of Presence, compostos por Edge Locations e
Regional Caches, distribuídos em mais de 90 cidades e 40 países.

Esses pontos permitem a entrega de conteúdo aos usuários finais com menor
latência, garantindo desempenho mais rápido e eficiente.