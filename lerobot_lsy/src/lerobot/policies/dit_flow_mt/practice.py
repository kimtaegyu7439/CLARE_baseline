class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, t, cond):
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = modulate(x, shift, scale)                         # (T, B, 512)
        x = self.linear(x)
        return x

class _TransformerDecoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(base_module) for _ in range(num_layers)])

        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, src, t, cond):
        x = src # [T, B, H]
        for layer in self.layers:
            x = layer(x, t, cond)
        return x

class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]
    
    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)

class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        
        return x * (1 + self.scale(c)[None]) + self.shift(c)[None]

    def reset_parameters(self):
        # DiT 논문의 "zero-init adaLN". 이걸 호출하는 곳은 _DiTDecoder.reset_parameters,
        # 그리고 그걸 호출하는 곳은 _TransformerDecoder.__init__이다(레이어 복제 직후).
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)

class _DiTDecoder(nn.Moudule):
    def __init__(self, d_model=256, nhead=6, dim_feedforward=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout) # (T, B, H)

        if NAMING_AS_MLP:
            self.mlp = MLP(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.activation = _get_activation_fn(activation_)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)

        self.attn_modulate = _ShiftScaleMod(d_model)
        self.attn_gate = _ZeroScaleMod(d_model)
        self.mlp_modulate = _ShiftScaleMod(d_model)
        self.mlp_gate = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):

        cond = cond + t
        
        x2 = self.attn_modulate(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weight=False)
        x = x + self.attn_gate(self.dropout1(x2), cond)
        x3 = self.mlp_modulate(self.norm2(x), cond)

        if NAMING_AS_MLP:
            x3 = self.mlp(x3)
        else:
            x3 = self.activation(self.linear1(x3))
            x3 = self.dropout2(x3)
            x3 = self.linear2(x3)
            x3 = self.dropout3(x3)
        
        x3 = self.mlp_gate(x3, cond)
        return x + x3

    def reset_parameters(self):
        # _TransformerDecoder.__init__에서 레이어를 deepcopy한 직후 호출된다.
        # 1) 2차원 이상 파라미터(= 모든 Linear/attn weight)를 xavier로 재초기화.
        #    LayerNorm weight와 각종 bias는 1차원이라 건드리지 않는다.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 2) 그다음 변조 레이어만 0으로 덮어쓴다(순서 중요 — 위 xavier를 되돌리는 것).
        #    결과적으로 학습 시작 시점의 블록은 "조건 무시 + residual 통과"에 가깝다.
        for s in (self.attn_modulate, self.attn_gate, self.mlp_modulate, self.mlp_gate):
            s.reset_parameters()

class _TimeNetwork(nn.Module):
    def __init__(self, frequency_embedding_dim, hidden_dim, leanrnable_w=False, max_period=1000):
        assert frequency_embedding_dim % 2 == 0
        half_dim = frequency_embedding_dim // 2
        super.__init__()
        
        w = np.log(max_period) / (half_dim-1)
        w = torch.exp(torch.arange(half_dim) * -w).float()

        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))
        self.out_net = nn.Sequential(nn.Linear(frequency_embedding_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, t):
        t = t[:, None] * self.w[None, :] # B,1 * 1,128 -> [B, 128]
        
        t = torch.cat((torch.cos(t), torch.sin(t)), dim=-1)
        return self.out_net(t)

class _DiTNoiseNet(nn.Moudule):
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        cond_dim,
        time_dim=256,
        hidden_dim=256,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
        clip_sample=False,
        clip_sample_range=1.0,
    ):
    
        super().__init__()
        self.ac_dim, self.ac_chunk = ac_dim, ac_chunk # 7, 16
        
        self.register_paremeter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True)
        )
        nn.init.xavier_uniform_(self.dec_pos)

        self.time_net = _TimeNetwork(time_dim, hidden_dim) # (B,) -> (B, 512)

        self.ac_proj = nn.Sequential(
            nn.Linear(ac.dim, ac.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )

        # (B, cond_dim=2576) -> (B, hidden_dim=512): 언어/상태/이미지 데이터가 여기서 처음 섞임.
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

    def forward(self, noisy_actions, time, global_cond):

        c = self.cond_proj(global_cond) # (B, 2576) -> (B, 512)
        time_enc = self.time_net(time) # (B,) -> (B, 512)
    
        ac_tokens = self.ac_proj(noisy_actions) # (B, T, 7) -> (B, T, 512)
        ac_tokens = ac_tokens.transpose(0, 1) # [B, T, 512] -> [T, B, 512]

        dec_in = ac_tokens + self.dec_pos[: ac_tokens.size(0)] #[T, B, 512]

        dec_out = self.decoder(dec_in, time_enc, c) 

        eps_out = self.eps_out(dec_out, time_enc, c) # [T, B, 7]
        return eps_out.transpose(0, 1) # [B, T, 7]

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, timesteps: int = 100, generator: torch.Generator | None = None
    ) -> torch.Tensor

        batch_size, device = condition.shape[0], condition.device
        x_0 = self.sample_noise(batch_size, device, generator)
        dt = 1.0 / timestpes
        
        t_all = (
            torch.arange(timesteps, device=device).float().unsqueeze(0).expand(batch_size, timesteps)
            / timesteps
        )

        for k in range(timesteps):
            t = t_all[:, k]
            x_0 = x_0 + dt * self.forward(x_0, t, condition)
            if self.clip_sample:
                x_0 = torch.clamp(x_0, -self.clip_sample_range, self.clip_sample_range)
        return x_0

    def sample_noise(self, batch_size: int, device, generator:torch.Generator | None=None) -> torch.Tensor:
        return torch.randn(batch_size, self.ac_chunk, self.ac_dim, deviec=device, generator=generator)

#------------------여기부터 LanuageEncoder

class LanguageEncoder(nn.Module):
    def __init__(self, config:DiTFlowMTConfig):
        super().__init__()

        self.config = config

        # from_pretrained가 HF Hub에서 받아옴. (저장위치는 HF_HUB_CACHE환경변수)
        self.tokenizer = CLIPTokenizer.from_pretrained(config.language_model_name)
        self.clip_model = CLIPTextModel.from_pretrained(config.language_model_name)

        self.cache = {}
        self.hidden_size = self.clip_model.config.hidden_size
        # clip-vit-base-patch32 -> 512

        if config.freeze_language_pretrained:
            self.clip_model.requires_grad_(False)
            self.clip_model.eval()

    def forward(self, texts):
        
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self.cache:
                cached_embeddings.append(self.cache[text])
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            inputs = self.tokenizer(
                uncached_texts,
                padding=True,
                truncation=True,
                return_tensor="pt",
                max_length=self.tokenizer.model_max_length
            )

            device = self.config.device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.set_grad_enabled(not self.clip_model.training):
                outputs = self.clip_model(**inputs)

            uncached_embeddings = outputs.pooler_output

            for text, embedding in zip(uncached_texts, uncached_embeddings):
                self.cache[text] = embedding.detach().cpu()
            
        all_embedding = [None] * len(texts)

        if uncached_texts:
            for i, emb in zip(uncached_indices, uncached_embeddings):
                all_embeddings[i] = emb
        
        for i, text in enumerate(texts):
            if text in self.cache and all_embeddings[i] is None:
                all_embeddings[i] = self.cache[text].to(self.config.device)

        return torch.stack(all_embeddings)

#------------------여기부터 ImageEncoder

class DINOv2Encoder(nn.Module):
    def __init__(self, config: DiTFlowMTConfig):
        super().__init__()
        self.config=config
        self._model = AutoModel.from_pretrained(config.vit_name)
        self._model.to(config.device)

        self._model.requires_grad_(False)
        self._model.eval()

        self._hidden_size = self._model.config.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, H, W] - 이미 ImageNet mean/std로 정규화된 상태로 들어옴
        outputs = self._model(x)
        cls_token = outputs.pooler_output # [B, 768]

        return cls_token


class DiTFLowModel(nn.Module):
    def __init__(self, config: DiTFlowMTConfig):
        super().__init__()
        self.config = config

        self.language_encoder = LanguageEncoder(config).to(self.config.device)

        self.language_embedding_projection = nn.Linear(self.language_encoder.hidden_size, config.hidden_dim) # 512 -> 512
        # 언어는 시간축이 없기 때문에 n_obs_steps를 곱하지 않고 따로 더해진다. (에피소드 내내 고정)
        language_cond_dim = condig.hidden_dim # 512
        
        if USE_STATE_PROJ:
            global_cond_dim = cofig.hidden_dim # 512
            self.state_proj = nn.Linear(self.config.rogot_state_feature.shape[0], config.hidden_dim)
        else:
            global_cond_dim = self.config.robot_state_feature_shape[0] # 여기만 씀

        
        # -- 이미지 --

        if self.config.image_features:
            self.pretrained_rgb_encoder = DINOv2Encoder(config)
            self.rgb_embedding_projection = nn.Linear(self.pretrained_rgb_encoder.hidden_size, config.hidden_dim)
            global_cond_dim += config.hidden_dim * len(self.config.image_features)

        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]
            
        self.velocity_net = _DiTNoiseNet(
            ac_dim=config.action_feature.shape[0],
            ac_chunk=config.horizon,
            cond_dim=config.frequency_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            dim_feedforward=config.dim_feedforward,
            nhead=config.num_heads,
            activation=config.activation
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
        )

        self.num_inference_steps = config.num_inference_steps or 100
        self.training_noise_sample = config.training_noise_sample

        if config.training_noise_sampling == "uniform":
            self.noise_distribution = torch.distributions.Uniform(
                low=0,
                high=1,
            )
        elif config.training_noise_sample == "beta":
            s = 0.999  # constant from the paper
            beta_dist = torch.distributions.Beta(
                concentration1=1.5,  # alpha
                concentration0=1.0,  # beta
            )
            affine_transform = torch.distributions.transforms.AffineTransform(loc=s, scale=-s)
            self.noise_distribution = torch.distributions.TransformedDistribution(
                beta_dist, [affine_transform]
            )

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: torch.Tensor | None=None,
        generator: torch.Generator | None=None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        if global_cond is not None:
            global_cond = global_cond.expand(batch_size, -1).to(device=device, dtype=dtype)

        sample = self.velocity_net.sample(
            global_cond, timesteps=self.num_inference_steps, generator=generator
        )
        return sample

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        with torch.no_grad():
            language_embedding = self.language_embedding(batch["task"])
        language_cond_feats = self.language_embedding_projection(language_embedding)
        
        global_cond_feats = [language_cond_feats]
        if USE_STATE_PROJ:
            states = einops.rearrange(batch[OBS_ROBOT], "b s ... -> (b, s)", b=batch_size, s=n_obs_steps)
            

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps
        